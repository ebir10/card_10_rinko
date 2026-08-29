"""誤答した画像1枚について、中間処理(カード抽出→コーナー→ランク文字→
テンプレート照合)を可視化し、原因を切り分けるための診断スクリプト。

出力先: debug_out/<画像名>/ に以下を保存する。
    00_original.png      元画像
    01_card.png           extract_card() 後の正立カード
    02_corner.png         左上コーナーのカラー切り出し
    03_corner_binary.png  コーナーの二値化(ここで行分割前の全輪郭が見える)
    04_glyph.png          実際にテンプレート照合に使われたランク文字(2値, 拡大)
    05_template_true.png  正解ランクのテンプレート(拡大)
    06_template_pred.png  予測されたランクのテンプレート(拡大)
    07_side_by_side.png   04/05/06 を横並びにした比較画像
また、全13ランクとの類似度スコアを降順で標準出力に表示する。

使い方:
    python tools/debug_misclassified.py C08 08
    python tools/debug_misclassified.py CK K
    (第1引数: data/deck/ 内のファイル名の拡張子抜き, 第2引数: 正解ランク)
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import classify
from common import DEFAULT_DECK_DIR


def upscale(img: np.ndarray, scale: int = 6) -> np.ndarray:
    return cv2.resize(img, (img.shape[1] * scale, img.shape[0] * scale), interpolation=cv2.INTER_NEAREST)


def to_bgr(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def debug_one(stem: str, true_rank: str, out_root: Path) -> None:
    path = DEFAULT_DECK_DIR / f"{stem}.png"
    out_dir = out_root / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"エラー: 読み込めません: {path}")
        return
    cv2.imwrite(str(out_dir / "00_original.png"), img)

    card = classify.extract_card(img)
    cv2.imwrite(str(out_dir / "01_card.png"), card)

    h, w = card.shape[:2]
    corner = card[0:int(h * classify.CORNER_H_FRAC), 0:int(w * classify.CORNER_W_FRAC)]
    cv2.imwrite(str(out_dir / "02_corner.png"), upscale(corner, 3))

    gray = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
    binary = classify._binarize_ink(gray)
    cv2.imwrite(str(out_dir / "03_corner_binary.png"), upscale(binary, 3))

    rows = classify._glyph_rows(binary)
    print(f"\n=== {stem} (正解: {true_rank}) ===")
    print(f"検出された行数: {len(rows)}  (各行のボックス数: {[len(r) for r in rows]})")
    for i, row in enumerate(rows):
        for box in row:
            x, y, bw, bh = box
            print(f"  行{i}: x={x} y={y} w={bw} h={bh} area={bw*bh}")

    glyph = classify.extract_rank_glyph(card)
    cv2.imwrite(str(out_dir / "04_glyph.png"), upscale(glyph))

    templates = classify._get_templates()
    scores = []
    for rank, template in templates.items():
        score = float(cv2.matchTemplate(glyph, template, cv2.TM_CCOEFF_NORMED)[0, 0])
        scores.append((rank, score))
    scores.sort(key=lambda kv: kv[1], reverse=True)

    print("類似度スコア(降順):")
    for rank, score in scores:
        mark = " <- 正解" if rank == true_rank else (" <- 予測(1位)" if (rank, score) == scores[0] else "")
        print(f"  {rank:>3}: {score:+.4f}{mark}")

    pred_rank = scores[0][0]
    if true_rank in templates:
        cv2.imwrite(str(out_dir / "05_template_true.png"), upscale(templates[true_rank]))
    if pred_rank in templates:
        cv2.imwrite(str(out_dir / "06_template_pred.png"), upscale(templates[pred_rank]))

    # 横並び比較画像: [抽出したグリフ] [正解テンプレート] [予測テンプレート]
    tiles = [to_bgr(upscale(glyph))]
    labels = [f"glyph({stem})"]
    if true_rank in templates:
        tiles.append(to_bgr(upscale(templates[true_rank])))
        labels.append(f"template[{true_rank}]=true")
    if pred_rank in templates:
        tiles.append(to_bgr(upscale(templates[pred_rank])))
        labels.append(f"template[{pred_rank}]=pred")

    pad = 20
    label_h = 30
    max_h = max(t.shape[0] for t in tiles)
    canvas_w = sum(t.shape[1] for t in tiles) + pad * (len(tiles) + 1)
    canvas = np.full((max_h + label_h + pad * 2, canvas_w, 3), 255, dtype=np.uint8)
    x = pad
    for tile, label in zip(tiles, labels):
        canvas[pad:pad + tile.shape[0], x:x + tile.shape[1]] = tile
        cv2.putText(
            canvas, label, (x, pad + max_h + label_h - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA,
        )
        x += tile.shape[1] + pad
    cv2.imwrite(str(out_dir / "07_side_by_side.png"), canvas)

    print(f"画像を保存しました: {out_dir}")


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) != 2:
        print("使い方: python tools/debug_misclassified.py <ファイル名(拡張子抜き)> <正解ランク>")
        return 1
    stem, true_rank = argv
    out_root = Path(__file__).resolve().parent.parent / "debug_out"
    debug_one(stem, true_rank, out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
