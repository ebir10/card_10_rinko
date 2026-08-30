"""全条件(通常/傾き/照明, 合成・実写含む)×両手法で評価し直し、
正解ラベルと異なる予測になった画像を results/misclassified/ に保存する。

出力(--target rank, 既定):
    results/misclassified/<condition>/<engine>/<file>_true<T>_pred<P>.png
        誤答した元画像に "true=T pred=P" を焼き込んだもの
        (pred が None の場合は "UNKNOWN" と表示)
    results/misclassified/<condition>/<engine>_grid.png
        その条件・手法の誤答をまとめて一覧できるグリッド画像
        (誤答が0件の組み合わせは生成しない)

--target suit を指定すると同じ形式で results/misclassified_suit/ 以下に
スート判別の誤答を保存する(ランク側の出力には触れない)。

条件・フォルダの対応は tools/report_stress_tests.py の CONDITIONS と揃えている。

使い方:
    python tools/collect_misclassified.py
    python tools/collect_misclassified.py --target suit
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import UNKNOWN, load_dataset
from report_stress_tests import CONDITIONS  # (key, label_ja, json_path, deck_dir) を使い回す

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT_BY_TARGET = {
    "rank": PROJECT_ROOT / "results" / "misclassified",
    "suit": PROJECT_ROOT / "results" / "misclassified_suit",
}

ENGINES = [
    ("classical", "古典的CV", "classify"),
    ("yolo", "YOLOv8", "classify_yolo"),
]


def annotate(img, true_rank: str, pred_rank: str) -> "cv2.Mat":
    banner_h = 34
    h, w = img.shape[:2]
    out = cv2.copyMakeBorder(img, banner_h, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(
        out, f"true={true_rank} pred={pred_rank}", (4, banner_h - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 200), 1, cv2.LINE_AA,
    )
    return out


def make_grid(entries: list[tuple[str, "cv2.Mat"]], title: str, out_path: Path) -> None:  # noqa: F821
    n = len(entries)
    cols = min(6, n)
    rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.6))
    axes = axes.reshape(rows, cols) if n > 1 else [[axes]]
    for i in range(rows * cols):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if i < n:
            label, img = entries[i]
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(label, fontsize=9)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", choices=("rank", "suit"), default="rank", help="評価対象: rank(既定) / suit")
    args = parser.parse_args(argv)
    out_root = OUT_ROOT_BY_TARGET[args.target]

    import classify
    import classify_yolo

    if args.target == "rank":
        predict_of = {"classify": classify.predict_rank, "classify_yolo": classify_yolo.predict_rank}
        true_label_of = lambda item: item.rank  # noqa: E731
    else:
        predict_of = {"classify": classify.predict_suit, "classify_yolo": classify_yolo.predict_suit}
        true_label_of = lambda item: item.suit  # noqa: E731

    total_wrong = 0
    summary_lines = []

    for key, label_ja, _json_path, deck_dir in CONDITIONS:
        if not deck_dir.is_dir():
            continue
        items = load_dataset(deck_dir)

        for engine_key, engine_ja, module_name in ENGINES:
            predict_fn = predict_of[module_name]
            wrong_entries: list[tuple[str, "cv2.Mat"]] = []  # noqa: F821
            out_dir = out_root / key / engine_key
            for item in items:
                true_label = true_label_of(item)
                pred = predict_fn(item.path)
                pred_label = pred if pred is not None else UNKNOWN
                if pred_label == true_label:
                    continue
                img = cv2.imread(str(item.path), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                annotated = annotate(img, true_label, pred_label)
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{item.path.stem}_true{true_label}_pred{pred_label}.png"
                cv2.imwrite(str(out_path), annotated)
                wrong_entries.append((f"{item.path.stem}\ntrue={true_label} pred={pred_label}", annotated))

            n_wrong = len(wrong_entries)
            total_wrong += n_wrong
            summary_lines.append(f"{label_ja:<14} x {engine_ja:<8}: 誤答 {n_wrong}/{len(items)}")
            print(summary_lines[-1])

            if wrong_entries:
                grid_path = out_root / key / f"{engine_key}_grid.png"
                make_grid(wrong_entries, f"{label_ja} x {engine_ja}: 誤答 {n_wrong}/{len(items)}", grid_path)
                print(f"  -> {out_dir} ({n_wrong}枚), {grid_path}")

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.txt").write_text("\n".join(summary_lines) + f"\n\n合計誤答数: {total_wrong}\n", encoding="utf-8")
    print(f"\n合計誤答数: {total_wrong}")
    print(f"保存先: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
