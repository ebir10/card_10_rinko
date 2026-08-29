"""E(data/deck/)から、難条件を模した2種類のテストセットを合成生成する。

    data/deck_tilt/   … カードを傾けて撮影した状況を模す(画像全体を回転)
    data/deck_light/  … 照明条件が違う状況を模す(明るさ/ガンマを変える)

新しい写真を撮る代わりに、既存のE(52枚)を画像処理で加工して作る。
ファイル名(=正解ラベル)はそのまま維持するので、tools/evaluate.py で
通常のE同様に評価できる(--deck-dir data/deck_tilt / data/deck_light)。
T(data/template_src/)は変更しない(テンプレートは通常条件のまま、
「通常条件で作ったテンプレートが難条件のカードにどこまで通用するか」を見る)。

乱数シードを固定しているため、再実行しても同じ画像が生成される。

使い方:
    python tools/make_stress_tests.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import DEFAULT_DECK_DIR, PROJECT_ROOT, load_dataset

SEED = 42
TILT_DIR = PROJECT_ROOT / "data" / "deck_tilt"
LIGHT_DIR = PROJECT_ROOT / "data" / "deck_light"

# 傾き: ±10°〜±22°の範囲でランダム(0°付近を避け、必ず有意な傾きを与える)
TILT_ANGLE_RANGE = (10.0, 22.0)

# 照明: 暗め(露出不足)か明るめ(露出過多)のどちらかをランダムに割り当てる
DARK_FACTOR_RANGE = (0.30, 0.50)
BRIGHT_FACTOR_RANGE = (1.70, 2.10)
GAMMA_JITTER = (0.85, 1.15)


def rotate_full(img: np.ndarray, angle: float) -> np.ndarray:
    """画像を切り取らずに回転させる(はみ出す分だけキャンバスを拡げる)。

    背景が黒クロスのため、拡げた余白は黒で塗る(=実際の背景色に馴染む)。
    """
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2
    m = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    m[0, 2] += (new_w / 2) - cx
    m[1, 2] += (new_h / 2) - cy
    return cv2.warpAffine(img, m, (new_w, new_h), borderValue=(0, 0, 0))


def relight(img: np.ndarray, factor: float, gamma: float) -> np.ndarray:
    """明るさ(factor倍)とガンマ補正で照明条件の違いを模す。"""
    img_f = img.astype(np.float32) / 255.0
    img_f = np.clip(img_f * factor, 0.0, 1.0)
    img_f = np.power(img_f, gamma)
    return (np.clip(img_f, 0.0, 1.0) * 255).astype(np.uint8)


def main() -> int:
    if not DEFAULT_DECK_DIR.is_dir():
        print(f"エラー: E(評価用画像フォルダ)が見つかりません: {DEFAULT_DECK_DIR}", file=sys.stderr)
        return 1

    items = load_dataset(DEFAULT_DECK_DIR)
    TILT_DIR.mkdir(parents=True, exist_ok=True)
    LIGHT_DIR.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)

    for item in items:
        img = cv2.imread(str(item.path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  [警告] 読み込めません: {item.path}")
            continue

        # --- 傾き ---
        sign = rng.choice([-1, 1])
        angle = sign * rng.uniform(*TILT_ANGLE_RANGE)
        tilted = rotate_full(img, angle)
        cv2.imwrite(str(TILT_DIR / item.path.name), tilted)

        # --- 照明 ---
        is_dark = rng.random() < 0.5
        factor = rng.uniform(*DARK_FACTOR_RANGE) if is_dark else rng.uniform(*BRIGHT_FACTOR_RANGE)
        gamma = rng.uniform(*GAMMA_JITTER)
        relit = relight(img, factor, gamma)
        cv2.imwrite(str(LIGHT_DIR / item.path.name), relit)

    print(f"完了: {len(items)}枚ずつ生成しました。")
    print(f"  傾き: {TILT_DIR}")
    print(f"  照明: {LIGHT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
