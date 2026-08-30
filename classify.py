"""OpenCVによる古典的な画像処理(テンプレートマッチング)でカードのランク・
スートを判別するモジュール。

公開インターフェースは2関数:

    def predict_rank(image_path: Path) -> str | None
    def predict_suit(image_path: Path) -> str | None

predict_rank は classify_yolo.py と全く同じシグネチャにすることで、
tools/evaluate.py を1本で共通化し、比較の公平性をコードレベルで保証する
(この契約はスート対応を追加した後も変更していない)。

テンプレートは data/template_src/ (T セット) から作る。
data/deck/ (E セット、評価専用)は一切参照しない。

流れ:
    1. 背景を輝度(明度)で分離する。背景は暗色(黒クロス等)、カードは
       明色という前提でグレースケール化しOtsu二値化する
    2. 残った最大の輪郭をカードとみなし、minAreaRect + 射影変換で
       正立・正規サイズの画像に補正する
    3. 左上コーナー(ランクが上段・スートマークが下段というトランプ共通の
       レイアウト)から、輪郭のy位置クラスタリングで「行」に分ける。
       行0をランク文字、行1をスートマークとして扱う(実際に52枚で
       確認した限り、通常はこの2行にきれいに分かれる。まれに行同士が
       融合して1行になることがあり、その場合スートは検出不能=Noneになる)。
    4. T セットから作った「ランク/スートごとのテンプレート画像」との
       正規化相互相関(cv2.matchTemplate)が最大のものを予測する
       (テンプレート作成そのものは build_templates()/build_suit_templates()
       が共通の _build_glyph_templates() を呼ぶ形に統一している)

カードやランク文字/スートマークを検出できなかった場合は None(=UNKNOWN)を
返す。これは classify_yolo.py が検出0件のときに None を返すのと同じ扱いで
あり、tools/evaluate.py はこの2つを区別せず同一のUNKNOWN扱いで集計する。
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from common import CardImage, DEFAULT_TEMPLATE_DIR, load_dataset

CARD_W, CARD_H = 400, 560          # カードを射影変換した後の正規サイズ
CORNER_W_FRAC = 0.17               # コーナー領域の幅(カード幅に対する割合)
CORNER_H_FRAC = 0.30               # コーナー領域の高さ
MIN_GLYPH_AREA = 20                # 文字とみなす輪郭の最小面積(px)
MAX_GLYPH_HEIGHT_FRAC = 0.6        # 文字とみなす輪郭の最大高さ(コーナー高さに対する割合)。
                                    # 絵札の枠線などコーナーの縦幅いっぱいに伸びる装飾線を除外する
ROW_GAP_TOL = 6                    # 同じ行とみなすy方向の許容ギャップ(px)
GLYPH_SIZE = (48, 64)              # ランクをテンプレート化する際の統一サイズ (w,h)。数字/文字は縦長。
SUIT_GLYPH_SIZE = (48, 48)         # スートマークをテンプレート化する際の統一サイズ (w,h)。
                                    # ハート/ダイヤ/クラブ/スペードは実測でおよそ正方形に近いため
                                    # ランクとは別サイズにしている。

RANK_ROW_INDEX = 0                 # コーナー内の行のうち、ランク文字とみなす行番号
SUIT_ROW_INDEX = 1                 # 同、スートマークとみなす行番号

_TEMPLATE_DIR: Path = DEFAULT_TEMPLATE_DIR
_TEMPLATE_CACHE: dict[str, np.ndarray] | None = None
_SUIT_TEMPLATE_CACHE: dict[str, np.ndarray] | None = None


class CardNotFoundError(RuntimeError):
    """画像からカードやランク文字を検出できなかったときの例外(内部用)。"""


# --- カード検出・射影変換 ----------------------------------------------------


def _order_points(pts: np.ndarray) -> np.ndarray:
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _find_card_contour(img: np.ndarray) -> np.ndarray:
    """暗い背景(黒クロス等)から明るいカードをOtsu二値化で切り出す。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, card_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    k = max(3, round(min(img.shape[:2]) * 0.01))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    card_mask = cv2.morphologyEx(card_mask, cv2.MORPH_OPEN, kernel)
    card_mask = cv2.morphologyEx(card_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(card_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise CardNotFoundError("カードらしき領域が見つかりません")
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < img.shape[0] * img.shape[1] * 0.03:
        raise CardNotFoundError("検出領域が小さすぎます(背景除去に失敗した可能性)")
    return contour


def extract_card(img: np.ndarray) -> np.ndarray:
    """写真からカード部分だけを射影変換で切り出し、正規サイズに揃える。

    90°/180°の向き違いまでは補正しない(カードは常に正立で撮影されている前提)。
    """
    contour = _find_card_contour(img)
    (cx, cy), (rw, rh), angle = cv2.minAreaRect(contour)
    # 背景との境界にアンチエイリアシングや影の縁が残りやすいので、矩形を
    # 内側に少し縮めてから射影変換する(warp後の背景の縁の写り込みを防ぐ)。
    shrink = 0.96
    rect = ((cx, cy), (rw * shrink, rh * shrink), angle)
    box = cv2.boxPoints(rect)
    src = _order_points(box)

    w = max(np.linalg.norm(src[0] - src[1]), np.linalg.norm(src[3] - src[2]))
    h = max(np.linalg.norm(src[0] - src[3]), np.linalg.norm(src[1] - src[2]))
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, matrix, (int(w), int(h)))
    if warped.shape[1] > warped.shape[0]:  # 横長なら90°回して縦長に揃える
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    return cv2.resize(warped, (CARD_W, CARD_H), interpolation=cv2.INTER_AREA)


def _binarize_ink(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def _glyph_rows(binary: np.ndarray) -> list[list[tuple[int, int, int, int]]]:
    """2値画像内の文字輪郭を、上から順に「行」ごとにグルーピングする。"""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_h = binary.shape[0] * MAX_GLYPH_HEIGHT_FRAC
    all_boxes = (cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= MIN_GLYPH_AREA)
    boxes = [b for b in all_boxes if b[3] <= max_h]
    if not boxes:
        return []
    boxes.sort(key=lambda b: b[1])
    rows: list[list[tuple[int, int, int, int]]] = [[boxes[0]]]
    for box in boxes[1:]:
        prev_bottom = max(b[1] + b[3] for b in rows[-1])
        if box[1] <= prev_bottom + ROW_GAP_TOL:
            rows[-1].append(box)
        else:
            rows.append([box])
    return rows


def _normalize_glyph(
    binary: np.ndarray, boxes: list[tuple[int, int, int, int]], size: tuple[int, int] = GLYPH_SIZE
) -> np.ndarray:
    xs0 = min(b[0] for b in boxes)
    ys0 = min(b[1] for b in boxes)
    xs1 = max(b[0] + b[2] for b in boxes)
    ys1 = max(b[1] + b[3] for b in boxes)
    pad = 2
    xs0, ys0 = max(0, xs0 - pad), max(0, ys0 - pad)
    xs1 = min(binary.shape[1], xs1 + pad)
    ys1 = min(binary.shape[0], ys1 + pad)
    crop = binary[ys0:ys1, xs0:xs1]
    return cv2.resize(crop, size, interpolation=cv2.INTER_AREA)


def _corner_binary(card: np.ndarray) -> np.ndarray:
    """正立済みカードから左上コーナーを切り出し、二値化する(ランク/スート共通の前処理)。"""
    h, w = card.shape[:2]
    corner = card[0:int(h * CORNER_H_FRAC), 0:int(w * CORNER_W_FRAC)]
    gray = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
    return _binarize_ink(gray)


def _extract_corner_glyph(card: np.ndarray, row_index: int, size: tuple[int, int], what: str) -> np.ndarray:
    """コーナーの二値画像から、指定した行番号(0=ランク, 1=スート)の文字/マークを切り出す。"""
    binary = _corner_binary(card)
    rows = _glyph_rows(binary)
    if len(rows) <= row_index:
        raise CardNotFoundError(f"コーナーから{what}を検出できません(検出行数={len(rows)})")
    return _normalize_glyph(binary, rows[row_index], size)


def extract_rank_glyph(card: np.ndarray) -> np.ndarray:
    """正規化済みカード画像から、左上コーナーのランク文字(2値画像)を切り出す。"""
    return _extract_corner_glyph(card, RANK_ROW_INDEX, GLYPH_SIZE, "ランク文字")


def extract_suit_glyph(card: np.ndarray) -> np.ndarray:
    """正規化済みカード画像から、左上コーナーのスートマーク(2値画像)を切り出す。"""
    return _extract_corner_glyph(card, SUIT_ROW_INDEX, SUIT_GLYPH_SIZE, "スートマーク")


# --- テンプレート管理 --------------------------------------------------------


def configure(template_dir: Path) -> None:
    """テンプレート元フォルダ(T)を変更し、キャッシュ(ランク・スート両方)を破棄する。テスト/CLI用。"""
    global _TEMPLATE_DIR, _TEMPLATE_CACHE, _SUIT_TEMPLATE_CACHE
    _TEMPLATE_DIR = Path(template_dir)
    _TEMPLATE_CACHE = None
    _SUIT_TEMPLATE_CACHE = None


def _build_glyph_templates(
    template_dir: Path,
    extract_fn: Callable[[np.ndarray], np.ndarray],
    label_of: Callable[[CardImage], str],
    what: str,
) -> dict[str, np.ndarray]:
    """T セットから、ラベル(ランク or スート)ごとに文字/マーク画素の中央値を
    取ったテンプレートを作る共通処理。extract_fn/label_of を差し替えるだけで
    build_templates() (ランク) と build_suit_templates() (スート) の両方を
    この1本の実装でまかなう(テンプレート作成方法そのものは変えていない)。
    """
    items: list[CardImage] = load_dataset(template_dir)
    glyphs_by_label: dict[str, list[np.ndarray]] = {}
    skipped = 0
    for item in items:
        img = cv2.imread(str(item.path), cv2.IMREAD_COLOR)
        if img is None:
            skipped += 1
            continue
        try:
            card = extract_card(img)
            glyph = extract_fn(card)
        except CardNotFoundError:
            skipped += 1
            continue
        glyphs_by_label.setdefault(label_of(item), []).append(glyph)

    templates: dict[str, np.ndarray] = {}
    for label, glyphs in glyphs_by_label.items():
        stacked = np.stack(glyphs).astype(np.float32)
        med = np.median(stacked, axis=0)
        _, binary = cv2.threshold(med.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
        templates[label] = binary

    if skipped:
        print(f"  [警告] {what}テンプレート作成時に {skipped} 枚をスキップしました(カード検出失敗)")
    return templates


def build_templates(template_dir: Path) -> dict[str, np.ndarray]:
    """T セットから、ランクごとのテンプレートを作る。"""
    return _build_glyph_templates(template_dir, extract_rank_glyph, lambda item: item.rank, "ランク")


def build_suit_templates(template_dir: Path) -> dict[str, np.ndarray]:
    """T セットから、スートごとのテンプレートを作る。"""
    return _build_glyph_templates(template_dir, extract_suit_glyph, lambda item: item.suit, "スート")


def _get_templates() -> dict[str, np.ndarray]:
    global _TEMPLATE_CACHE
    if _TEMPLATE_CACHE is None:
        if not _TEMPLATE_DIR.is_dir():
            raise FileNotFoundError(
                f"テンプレート元フォルダが見つかりません: {_TEMPLATE_DIR}\n"
                "prepare_deck.py で撮影したT(テンプレート専用)画像をここに用意してください。"
            )
        _TEMPLATE_CACHE = build_templates(_TEMPLATE_DIR)
    return _TEMPLATE_CACHE


def _get_suit_templates() -> dict[str, np.ndarray]:
    global _SUIT_TEMPLATE_CACHE
    if _SUIT_TEMPLATE_CACHE is None:
        if not _TEMPLATE_DIR.is_dir():
            raise FileNotFoundError(
                f"テンプレート元フォルダが見つかりません: {_TEMPLATE_DIR}\n"
                "prepare_deck.py で撮影したT(テンプレート専用)画像をここに用意してください。"
            )
        _SUIT_TEMPLATE_CACHE = build_suit_templates(_TEMPLATE_DIR)
    return _SUIT_TEMPLATE_CACHE


# --- 公開インターフェース ----------------------------------------------------


def _best_match(glyph: np.ndarray, templates: dict[str, np.ndarray]) -> str | None:
    best_label, best_score = None, -1.0
    for label, template in templates.items():
        score = float(cv2.matchTemplate(glyph, template, cv2.TM_CCOEFF_NORMED)[0, 0])
        if score > best_score:
            best_label, best_score = label, score
    return best_label


def predict_rank(image_path: Path) -> str | None:
    """カード写真1枚のランクを判別する。判別できなければ None (UNKNOWN)。"""
    templates = _get_templates()
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    try:
        card = extract_card(img)
        glyph = extract_rank_glyph(card)
    except CardNotFoundError:
        return None
    return _best_match(glyph, templates)


def predict_suit(image_path: Path) -> str | None:
    """カード写真1枚のスート(C/S/D/H)を判別する。判別できなければ None (UNKNOWN)。

    predict_rank と同じ形の契約(classify_yolo.predict_suit と揃えている)。
    行0(ランク)と行1(スート)が融合してしまったカードなど、コーナーから
    スートマークを単独で切り出せない場合は None を返す。
    """
    templates = _get_suit_templates()
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    try:
        card = extract_card(img)
        glyph = extract_suit_glyph(card)
    except CardNotFoundError:
        return None
    return _best_match(glyph, templates)
