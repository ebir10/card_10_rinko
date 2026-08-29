"""難条件テスト(傾き/照明)の結果を、画像でひと目に分かるようにまとめる。

出力(results/stress_test/):
    samples_grid.png     … サンプルカードの [通常/傾き/照明] 3列比較
    accuracy_chart.png   … 3条件 x 2手法の正解率グループ棒グラフ
    summary_table.png    … accuracy/macroF1/UNKNOWN/推論msの一覧表(画像化)

使い方:
    python tools/make_stress_tests.py                 # data/deck_tilt, data/deck_light を生成
    python tools/evaluate.py --engine both --save-json results/comparison.json
    python tools/evaluate.py --engine both --deck-dir data/deck_tilt  --save-json results/stress_test/comparison_tilt.json
    python tools/evaluate.py --engine both --deck-dir data/deck_light --save-json results/stress_test/comparison_light.json
    python tools/report_stress_tests.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import PROJECT_ROOT

RESULTS_DIR = PROJECT_ROOT / "results"
STRESS_DIR = RESULTS_DIR / "stress_test"

CONDITIONS = [
    ("baseline", "通常(E)", RESULTS_DIR / "comparison.json", PROJECT_ROOT / "data" / "deck"),
    ("tilt", "傾き", STRESS_DIR / "comparison_tilt.json", PROJECT_ROOT / "data" / "deck_tilt"),
    ("light", "照明変化", STRESS_DIR / "comparison_light.json", PROJECT_ROOT / "data" / "deck_light"),
]
SAMPLE_FILES = ["H10.png", "C08.png", "SK.png"]  # サンプル比較に使うカード
ENGINE_JA = {
    "classify (OpenCV template matching)": "古典的CV",
    "classify_yolo": "YOLOv8",
}


def load_condition_results(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {item["name"]: item for item in payload}


def make_samples_grid() -> None:
    rows = len(SAMPLE_FILES)
    cols = len(CONDITIONS)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 3.4))
    for r, filename in enumerate(SAMPLE_FILES):
        for c, (_key, label_ja, _json_path, deck_dir) in enumerate(CONDITIONS):
            ax = axes[r][c]
            img_path = deck_dir / filename
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is not None:
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(label_ja, fontsize=13)
            if c == 0:
                ax.set_ylabel(filename.replace(".png", ""), fontsize=11)
    fig.suptitle("サンプル画像: 通常 / 傾き / 照明変化", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = STRESS_DIR / "samples_grid.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"保存: {out}")


def make_accuracy_chart(all_results: dict[str, dict[str, dict]]) -> None:
    engines = list(ENGINE_JA.keys())
    condition_labels = [label for _key, label, _p, _d in CONDITIONS]
    condition_keys = [key for key, _label, _p, _d in CONDITIONS]

    x = np.arange(len(condition_keys))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"classify (OpenCV template matching)": "#4c9f70", "classify_yolo": "#e4b73f"}

    for i, engine in enumerate(engines):
        accs = [all_results[key][engine]["accuracy"] * 100 for key in condition_keys]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, accs, width, label=ENGINE_JA[engine], color=colors.get(engine))
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f"{acc:.1f}%", ha="center", va="bottom", fontsize=10)

    ax.set_ylim(0, 110)
    ax.set_ylabel("ランク正解率 (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(condition_labels)
    ax.set_title("難条件による正解率の変化(全52枚)")
    ax.axhline(100, color="gray", linewidth=0.5, linestyle="--")
    ax.legend()
    fig.tight_layout()
    out = STRESS_DIR / "accuracy_chart.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"保存: {out}")


def make_summary_table(all_results: dict[str, dict[str, dict]]) -> None:
    engines = list(ENGINE_JA.keys())
    condition_labels = [label for _key, label, _p, _d in CONDITIONS]
    condition_keys = [key for key, _label, _p, _d in CONDITIONS]

    rows = []
    for key, label in zip(condition_keys, condition_labels):
        for engine in engines:
            r = all_results[key][engine]
            rows.append([
                label, ENGINE_JA[engine],
                f"{r['accuracy']*100:.1f}%", f"{r['macro_f1']:.3f}",
                str(r["n_unknown"]), f"{r['mean_inference_ms']:.1f} ms",
            ])

    col_labels = ["条件", "手法", "accuracy", "macroF1", "UNKNOWN", "推論時間/枚"]
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(rows) + 1))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)
    for c in range(len(col_labels)):
        table[0, c].set_facecolor("#2c4a35")
        table[0, c].set_text_props(color="white", weight="bold")
    ax.set_title("難条件比較 サマリー表(全52枚, 主指標=ランク正解率)", fontsize=13, pad=20)
    fig.tight_layout()
    out = STRESS_DIR / "summary_table.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"保存: {out}")


def main() -> int:
    STRESS_DIR.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, dict[str, dict]] = {}
    for key, _label, json_path, _deck_dir in CONDITIONS:
        if not json_path.exists():
            print(f"エラー: 結果ファイルが見つかりません: {json_path}", file=sys.stderr)
            return 1
        all_results[key] = load_condition_results(json_path)

    make_samples_grid()
    make_accuracy_chart(all_results)
    make_summary_table(all_results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
