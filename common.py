"""プロジェクト全体で共有する定数・データセット読み込みユーティリティ。

prepare_deck.py が作る各フォルダ(例: H10.png = ハートの10)を
「ファイル名 = 正解ラベル」の教師データ/評価データとして扱う。

このプロジェクトのデータ設計(3層構成ではなく2セット):
    T (data/template_src/) … classify.py がテンプレートを作る専用。評価には使わない。
    E (data/deck/)          … classify.py・classify_yolo.py 両方の評価専用、
                               かつ demo(10を作る)のランダム抽選元。
                               T の作成にも YOLO 側の学習にも使わない。
T と E は別の撮影セッションであること(理想は別日)。これが比較の公平性の土台になる。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SUITS: tuple[str, ...] = ("C", "S", "D", "H")
NUMBER_RANKS: tuple[str, ...] = ("A", "02", "03", "04", "05", "06", "07", "08", "09", "10")
FACE_RANKS: tuple[str, ...] = ("J", "Q", "K")
RANKS: tuple[str, ...] = NUMBER_RANKS + FACE_RANKS  # 13種類: A,02,...,10,J,Q,K

# 「10を作る」計算で使う数値。標準ルールに合わせ J=11, Q=12, K=13 とする。
RANK_VALUE: dict[str, int] = {
    "A": 1, "02": 2, "03": 3, "04": 4, "05": 5, "06": 6, "07": 7,
    "08": 8, "09": 9, "10": 10, "J": 11, "Q": 12, "K": 13,
}
SUIT_MARK: dict[str, str] = {"C": "♣", "S": "♠", "D": "♦", "H": "♥"}

UNKNOWN = "UNKNOWN"  # 判別不能(古典CVは検出失敗、YOLOは検出0件)を表す共通ラベル

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_TEMPLATE_DIR = PROJECT_ROOT / "data" / "template_src"  # T
DEFAULT_DECK_DIR = PROJECT_ROOT / "data" / "deck"              # E


@dataclass(frozen=True)
class CardImage:
    path: Path
    suit: str
    rank: str

    @property
    def label(self) -> str:
        return f"{self.rank}{SUIT_MARK.get(self.suit, self.suit)}"


def parse_filename(path: Path) -> tuple[str, str]:
    """'H10.png' -> ('H', '10') のようにファイル名を (スート, ランク) に分解する。"""
    stem = path.stem
    suit, rank = stem[0], stem[1:]
    if suit not in SUITS or rank not in RANKS:
        raise ValueError(f"想定外のファイル名です: {path.name}")
    return suit, rank


def load_dataset(image_dir: Path) -> list[CardImage]:
    """image_dir (prepare_deck.py の出力フォルダ) からラベル付き画像を読み込む。"""
    items = []
    for suit in SUITS:
        for rank in RANKS:
            path = image_dir / f"{suit}{rank}.png"
            if path.exists():
                items.append(CardImage(path, suit, rank))
    if not items:
        raise FileNotFoundError(f"{image_dir} にカード画像が見つかりません")
    return items
