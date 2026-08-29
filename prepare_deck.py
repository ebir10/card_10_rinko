"""撮影した写真を data/<dst>/ の命名規則へ一括変換する前処理スクリプト。

このプロジェクトでは同じスクリプトを**2回**使う:

    セッション1 (T: テンプレート作成専用) -> --dst data/template_src
    セッション2 (E: 両手法の評価+ランダム抽選専用) -> --dst data/deck

T と E は別の物理カード・別の撮影回であること。同じ日に続けて撮る場合も、
必ず一度カードを片付けて台紙とスマホの位置をリセットしてから2周目に入ること。
並べたまま連続で撮ると、ほぼ同じ画像になり分けた意味が薄れる。

スマホで下記の順に撮影した 52 枚を撮影順に並べ、正解ラベル兼ファイル名に変換する。

    【数札 40 枚】♣A〜♣10 → ♠A〜♠10 → ♦A〜♦10 → ♥A〜♥10
    【絵札 12 枚】♣J♣Q♣K → ♠J♠Q♠K → ♦J♦Q♦K → ♥J♥Q♥K

    CA.png, C02.png, ... , C10.png, SA.png, ... , H10.png,   ← ここまで 40 枚
    CJ.png, CQ.png, CK.png, SJ.png, ... , HK.png              ← 絵札 12 枚

初期版は数札 40 枚しか使わないため、40 枚を撮り終えた時点で --no-faces を付ければ
先にフォルダを作り始められる。同時に
  - EXIF の回転情報を適用して確定させる(以降の処理で向きがブレない)
  - 長辺を一定サイズに縮小する(処理速度対策)
  - PNG で保存し直す(再圧縮による劣化を防ぐ)
を行う。

並び順(=どの写真がどのカードか)の決め方は --order で選ぶ:
  exif  … EXIF の撮影日時順(既定)。同時刻はファイル名の自然順で解決
  name  … ファイル名の自然順。Windows のエクスプローラーの表示順と一致する
  mtime … ファイルの更新日時順(EXIF が無い場合の保険)

既定の exif 順で並べたとき、name 順と結果が食い違う場合は警告を出す。

デフォルトは **dry-run**。対応表を目視で確認してから --apply を付けて実行する。

使い方:
    python prepare_deck.py --src data/photos_template --dst data/template_src --apply
    python prepare_deck.py --src data/photos_eval --dst data/deck --apply
    python prepare_deck.py --src data/photos --dst data/deck --order name

必要なライブラリ: opencv-python, numpy, pillow
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# --- 定数 -----------------------------------------------------------------

SUITS: tuple[str, ...] = ("C", "S", "D", "H")  # ♣ ♠ ♦ ♥ の撮影順
NUMBER_RANKS: tuple[str, ...] = (
    "A", "02", "03", "04", "05", "06", "07", "08", "09", "10",
)
FACE_RANKS: tuple[str, ...] = ("J", "Q", "K")
LONG_SIDE_PX: int = 1200  # 縮小後の長辺
IMAGE_SUFFIXES: frozenset[str] = frozenset({".jpg", ".jpeg", ".png"})
EXIF_DATETIME_ORIGINAL: int = 36867  # EXIF タグ番号: DateTimeOriginal
EXIF_DATETIME_FORMAT: str = "%Y:%m:%d %H:%M:%S"
_DIGITS_RE = re.compile(r"(\d+)")


# --- 並び順の決定 ----------------------------------------------------------


def natural_key(name: str) -> tuple[object, ...]:
    """自然順ソート用のキー。数字部分を数値として比較する(IMG_9 < IMG_10)。"""
    parts = _DIGITS_RE.split(name.lower())
    return tuple(int(p) if p.isdigit() else p for p in parts)


def exif_datetime(path: Path) -> datetime | None:
    """EXIF の撮影日時(DateTimeOriginal)を返す。取得できなければ None。"""
    try:
        with Image.open(path) as img:
            exif = img.getexif()
    except Exception:
        return None
    raw = exif.get(EXIF_DATETIME_ORIGINAL)
    if not isinstance(raw, str):
        return None
    try:
        return datetime.strptime(raw, EXIF_DATETIME_FORMAT)
    except ValueError:
        return None


def collect_photos(src: Path) -> list[Path]:
    """src 直下の画像ファイルを列挙する(順序は未定)。"""
    return [p for p in src.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES]


def order_photos(photos: list[Path], order: str) -> tuple[list[Path], list[str]]:
    """指定した基準で写真を並べ替え、(並んだリスト, 警告メッセージ) を返す。"""
    warnings: list[str] = []
    by_name = sorted(photos, key=lambda p: natural_key(p.name))

    if order == "name":
        return by_name, warnings

    if order == "mtime":
        return sorted(photos, key=lambda p: (p.stat().st_mtime, natural_key(p.name))), warnings

    # order == "exif"
    stamps = {p: exif_datetime(p) for p in photos}
    missing = [p for p, dt in stamps.items() if dt is None]
    if missing:
        warnings.append(
            f"EXIF の撮影日時が読めない画像が {len(missing)} 件あります"
            f"(例: {missing[0].name})。ファイル名の自然順に切り替えます。"
        )
        return by_name, warnings

    ordered = sorted(photos, key=lambda p: (stamps[p], natural_key(p.name)))
    if [p.name for p in ordered] != [p.name for p in by_name]:
        warnings.append(
            "撮影日時順とファイル名順で並びが食い違っています。"
            "撮り直しの消し忘れ・無関係な画像の混入・連番の一周などが疑われます。"
            "対応表を必ず目視で確認してください。"
        )
    return ordered, warnings


# --- 画像処理 --------------------------------------------------------------


def expected_names(include_faces: bool = True) -> list[str]:
    """撮影順に並んだ正式ファイル名(拡張子なし)を返す。"""
    names = [f"{suit}{rank}" for suit in SUITS for rank in NUMBER_RANKS]
    if include_faces:
        names += [f"{suit}{rank}" for suit in SUITS for rank in FACE_RANKS]
    return names


def load_and_resize(path: Path, long_side: int = LONG_SIDE_PX) -> np.ndarray:
    """画像を読み込み、EXIF 回転を適用したうえで長辺を long_side に縮小する。"""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(
            f"画像を読み込めません: {path}\n"
            "HEIC 形式の可能性があります。iPhone の場合は "
            "[設定] > [カメラ] > [フォーマット] を『互換性優先』にして撮り直してください。"
        )
    h, w = img.shape[:2]
    scale = long_side / max(h, w)
    if scale >= 1.0:
        return img
    new_size = (round(w * scale), round(h * scale))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)


# --- CLI -------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True, help="撮影した写真のフォルダ")
    parser.add_argument("--dst", type=Path, required=True, help="出力先(data/template_src または data/deck)")
    parser.add_argument("--order", choices=("exif", "name", "mtime"), default="exif", help="並び順の基準(既定: exif)")
    parser.add_argument("--no-faces", action="store_true", help="絵札(J/Q/K)を含めず、数札 40 枚だけを処理する")
    parser.add_argument("--apply", action="store_true", help="実際に書き込む(省略時は dry-run)")
    parser.add_argument("--long-side", type=int, default=LONG_SIDE_PX, help="縮小後の長辺 px")
    args = parser.parse_args(argv)

    if not args.src.is_dir():
        print(f"エラー: フォルダが見つかりません: {args.src}", file=sys.stderr)
        return 1

    photos = collect_photos(args.src)
    ordered, warnings = order_photos(photos, args.order)
    names = expected_names(include_faces=not args.no_faces)

    print(f"入力: {len(ordered)} 枚 / 期待: {len(names)} 枚 / 並び順: {args.order}")
    for w in warnings:
        print(f"  [警告] {w}")

    if len(ordered) != len(names):
        print(
            "\nエラー: 枚数が一致しません。撮り漏れ・重複撮影・関係ないファイルの混入を確認してください。",
            file=sys.stderr,
        )
        for i, p in enumerate(ordered):
            print(f"  [{i:2d}] {p.name}")
        return 1

    n_numbers = len(SUITS) * len(NUMBER_RANKS)
    print("\n--- 対応表(並び順 → 正式名)---")
    for i, (photo, name) in enumerate(zip(ordered, names)):
        if i == n_numbers:
            print(f"  {'---- ここまで数札 ' + str(n_numbers) + ' 枚 / ここから絵札 ----':-^70}")
        dt = exif_datetime(photo)
        stamp = dt.strftime("%m/%d %H:%M:%S") if dt else "EXIF なし"
        print(f"  [{i:2d}] {photo.name:>28}  {stamp}  ->  {name}.png")

    if not args.apply:
        print("\ndry-run です。上の対応表が正しければ --apply を付けて再実行してください。")
        return 0

    args.dst.mkdir(parents=True, exist_ok=True)
    for photo, name in zip(ordered, names):
        img = load_and_resize(photo, args.long_side)
        out = args.dst / f"{name}.png"
        if not cv2.imwrite(str(out), img):
            raise OSError(f"書き込みに失敗しました: {out}")
        print(f"  書き出し: {out}  {img.shape[1]}x{img.shape[0]}")

    print(f"\n完了: {len(names)} 枚を {args.dst} に書き出しました。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
