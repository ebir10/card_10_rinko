"""既存の公開データセットで学習済みのYOLOv8物体検出モデルを使って
カードのランクを判別するモジュール。

    モデル: mustafakemal0146/playing-cards-yolov8 (Hugging Face)
    実体  : YOLOv8n を Roboflow Playing Cards Dataset (52クラス, CC0) で
            50エポック学習したもの。MIT License。
            https://huggingface.co/mustafakemal0146/playing-cards-yolov8
            https://universe.roboflow.com/augmented-startups/playing-cards-ow27d

重要: このURLは「データセット」ではなく「学習済みモデル」である。
学習は不要で、重みをダウンロードして推論するだけで使える。
このプロジェクトの52枚の写真(data/deck/, data/template_src/)では
一切学習しない。ゼロショット評価。

公開インターフェースは classify.py と全く同じシグネチャ:

    def predict_rank(image_path: Path) -> str | None

YOLO側の実装で注意した3点(すべてこの関数内で処理):
    1. クラス名の順序: このモデルのクラス名は '10C' '2H' 'AS' のように
       [ランク][スート] の順(スートが末尾)。このプロジェクトのファイル
       命名 'C10.png' 'H2.png' (スートが先頭)とは順序が逆なので、
       parse_class_name() で変換する。1桁の数字ランクは common.RANKS の
       命名(ゼロ埋め2桁, 例 '02')に合わせてゼロ埋めする。
    2. 1枚から複数検出: カードは左上と右下の2か所にランクの数字/文字が
       印字されているため、1枚の写真から2個の検出が返るのが普通。
       ここでは「最も信頼度(confidence)が高い検出を採用する」というルールを
       決めて実装する。左上・右下の平均を取る、両方一致した場合のみ採用する、
       といった代替ルールも考えられるが、実装の単純さとclassify.py側が
       「1枚→1個の判定」であることに合わせ、最高信頼度1個を採用する方式にした。
    3. 検出0件の扱い: classify.py がカード/文字を検出できないときに
       None を返すのと同じ扱いにするため、検出0件のときは None を返す。
       tools/evaluate.py はこれを共通の UNKNOWN として集計する。

必要ライブラリ: ultralytics (torchも自動で入る)
初回実行時に重み(約6MB)をHugging Faceからダウンロードする。
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

from ultralytics import YOLO

# --- 学習済み重み ------------------------------------------------------------

WEIGHTS_URL = (
    "https://huggingface.co/mustafakemal0146/playing-cards-yolov8/"
    "resolve/main/playing_cards_model_0_playing-cards-colab.pt"
)
WEIGHTS_PATH = Path(__file__).resolve().parent / "pretrained_models" / "playing-cards-yolov8.pt"

IMGSZ = 640      # このモデルの学習時の入力サイズ
CONF_THRESHOLD = 0.25  # 検出の信頼度しきい値(ultralyticsの一般的な既定値)

_model: YOLO | None = None


def _ensure_weights(path: Path = WEIGHTS_PATH, url: str = WEIGHTS_URL) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, str(path))
    return path


def _get_model() -> YOLO:
    global _model
    if _model is None:
        _ensure_weights()
        _model = YOLO(str(WEIGHTS_PATH))
    return _model


# --- クラス名の変換 ----------------------------------------------------------


def parse_class_name(name: str) -> tuple[str, str]:
    """'10C' -> ('C', '10'), 'AS' -> ('S', 'A') のように、このモデルの
    クラス名([ランク][スート]の順)を (スート, ランク) に分解する。
    1桁の数字ランクは common.RANKS の命名(ゼロ埋め2桁)に合わせる。
    """
    suit = name[-1]
    rank = name[:-1]
    if rank.isdigit() and len(rank) == 1:
        rank = f"0{rank}"
    return suit, rank


# --- 公開インターフェース ----------------------------------------------------


def predict_rank(image_path: Path) -> str | None:
    """カード写真1枚のランクを判別する。検出0件なら None (UNKNOWN)。

    1枚から複数検出(通常は左上・右下の2個)が返った場合は、
    最も信頼度(confidence)が高い検出のクラスを採用する。
    """
    model = _get_model()
    result = model.predict(source=str(image_path), imgsz=IMGSZ, conf=CONF_THRESHOLD, verbose=False)[0]

    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None

    best_idx = int(boxes.conf.argmax())  # 複数検出のうち最高信頼度を採用(注意点2)
    class_name = result.names[int(boxes.cls[best_idx])]
    _suit, rank = parse_class_name(class_name)
    return rank
