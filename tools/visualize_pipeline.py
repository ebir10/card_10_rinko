"""古典的CV(テンプレートマッチング)とYOLOv8、それぞれの画像処理の流れを
可視化し、`results/pipeline/` に画像として保存するだけのスクリプト。
Webアプリ(`webapp/`)には一切反映しない。

古典的CV側(`classify.py`)は、指定したサンプルカードについて以下の
段階をそれぞれ画像として出力する:

    00_original.png         元の写真(E, data/deck/)
    01_card_extracted.png   射影変換でカードを正立・正規サイズに補正
    02_corner_crop.png      左上コーナーの切り出し(カラー)
    03_grayscale.png        グレースケール変換
    04_binarized.png        Otsu二値化(インク=白、背景=黒)
    05_normalized_glyph.png 行分離してランク文字だけを抽出し、テンプレートと
                             同じサイズ(GLYPH_SIZE)に正規化したもの
    06_template.png         そのランクのテンプレート画像(Tから事前作成済み)

YOLOv8側(`classify_yolo.py`)は、検出結果(バウンディングボックス・
クラス名・信頼度を描画した画像)を出力する:

    00_original.png
    01_detection.png

使い方:
    python tools/visualize_pipeline.py                  # 既定のサンプル3枚
    python tools/visualize_pipeline.py --cards H10 SK CA D07
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import classify
from common import DEFAULT_DECK_DIR, PROJECT_ROOT, parse_filename

DEFAULT_CARDS = ["H10", "SK", "CA"]
OUT_ROOT = PROJECT_ROOT / "results" / "pipeline"


def upscale(img: np.ndarray, scale: int = 3) -> np.ndarray:
    return cv2.resize(img, (img.shape[1] * scale, img.shape[0] * scale), interpolation=cv2.INTER_NEAREST)


def visualize_classical(stem: str, out_dir: Path) -> None:
    path = DEFAULT_DECK_DIR / f"{stem}.png"
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"  [警告] 読み込めません: {path}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_dir / "00_original.png"), img)

    card = classify.extract_card(img)
    cv2.imwrite(str(out_dir / "01_card_extracted.png"), card)

    h, w = card.shape[:2]
    corner = card[0:int(h * classify.CORNER_H_FRAC), 0:int(w * classify.CORNER_W_FRAC)]
    cv2.imwrite(str(out_dir / "02_corner_crop.png"), upscale(corner))

    gray = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(str(out_dir / "03_grayscale.png"), upscale(gray))

    binary = classify._binarize_ink(gray)
    cv2.imwrite(str(out_dir / "04_binarized.png"), upscale(binary))

    glyph = classify.extract_rank_glyph(card)  # 行分離 + GLYPH_SIZE への正規化まで含む
    cv2.imwrite(str(out_dir / "05_normalized_glyph.png"), upscale(glyph, 6))

    templates = classify._get_templates()
    _suit, rank = parse_filename(path)
    if rank in templates:
        cv2.imwrite(str(out_dir / "06_template.png"), upscale(templates[rank], 6))
    else:
        print(f"  [警告] ランク {rank} のテンプレートが見つかりません")

    print(f"  古典的CV: {out_dir}")


def visualize_yolo(stem: str, out_dir: Path) -> None:
    try:
        import classify_yolo
    except ImportError as e:
        print(f"  [警告] classify_yolo を読み込めません: {e}")
        return

    path = DEFAULT_DECK_DIR / f"{stem}.png"
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"  [警告] 読み込めません: {path}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_dir / "00_original.png"), img)

    model = classify_yolo._get_model()
    result = model.predict(
        source=str(path), imgsz=classify_yolo.IMGSZ, conf=classify_yolo.CONF_THRESHOLD, verbose=False
    )[0]
    annotated = result.plot()  # バウンディングボックス・クラス名・信頼度を描画済みのBGR画像
    cv2.imwrite(str(out_dir / "01_detection.png"), annotated)

    print(f"  YOLOv8  : {out_dir}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cards", nargs="+", default=DEFAULT_CARDS, help="可視化するカード(拡張子抜きのファイル名、例: H10)")
    args = parser.parse_args(argv)

    for stem in args.cards:
        print(f"=== {stem} ===")
        visualize_classical(stem, OUT_ROOT / stem / "classical")
        visualize_yolo(stem, OUT_ROOT / stem / "yolo")

    print(f"\n完了: {OUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
