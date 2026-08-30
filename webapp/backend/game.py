"""Webアプリ用のゲームロジック: Eからランダムに4枚引いて判別し、10を作る式を探す。

classify.py / classify_yolo.py どちらも predict_rank(image_path) -> str | None
という同じ契約なので、エンジンの切り替えは importするモジュールを変えるだけ。
判別に使う数字は「実際に画像処理/YOLOで読み取った値」であり、正解ラベルでは
ない(=このプロジェクト全体のテーマ「画像処理で読み取った数字から10を導く」
に沿っている)。
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

# webapp/backend/ の2つ上(プロジェクトルート)を import パスに追加する
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from common import DEFAULT_DECK_DIR, PROJECT_ROOT, RANK_VALUE, load_dataset  # noqa: E402
import make_ten  # noqa: E402

# 撮影条件 -> 画像フォルダ。tilt/light はセッション3・4で実際に撮影した
# 実写データ(data/deck_tilt_photo/, data/deck_light_photo/, ファイル名・
# ラベルはEと共通の52枚)。合成版(tools/make_stress_tests.py生成)は
# data/deck_tilt/, data/deck_light/ にそのまま残しており、COMPARE.md §9/§10
# の比較記録として参照できる。
CONDITION_DIRS: dict[str, Path] = {
    "normal": DEFAULT_DECK_DIR,
    "tilt": PROJECT_ROOT / "data" / "deck_tilt_photo",
    "light": PROJECT_ROOT / "data" / "deck_light_photo",
}


def draw_and_solve(
    engine: str = "yolo",
    count: int = 4,
    target: int = 10,
    seed: int | None = None,
    condition: str = "normal",
) -> dict:
    """指定した撮影条件のフォルダからcount枚引き、engineで判別し、targetを作る式を探す。"""
    if engine == "classical":
        import classify as recognizer
    else:
        import classify_yolo as recognizer

    deck_dir = CONDITION_DIRS.get(condition, DEFAULT_DECK_DIR)
    items = load_dataset(deck_dir)
    rng = random.Random(seed)
    drawn = rng.sample(items, min(count, len(items)))

    cards = []
    values = []
    for item in drawn:
        pred = recognizer.predict_rank(item.path)
        cards.append({
            "filename": item.path.name,
            "pred_rank": pred,  # None ならUNKNOWN(判別不能)
            "pred_value": RANK_VALUE.get(pred) if pred else None,
        })
        if pred is not None and pred in RANK_VALUE:
            values.append(RANK_VALUE[pred])

    solvable = len(values) == len(drawn)
    expression = make_ten.solve(values, target) if solvable else None

    return {
        "engine": engine,
        "condition": condition,
        "target": target,
        "cards": cards,
        "values": values,
        "unknown_count": len(drawn) - len(values),
        "expression": expression,
    }
