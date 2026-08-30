"""スート(C/S/D/H)判別の結果を、tools/report_stress_tests.py と同じ
5条件×2手法の枠組みでグラフ・表にまとめる。

出力(results/stress_test/):
    accuracy_chart_suit.png … 3条件x2手法(合成/実写含め5条件)の正解率グループ棒グラフ
    summary_table_suit.png  … accuracy/macroF1/UNKNOWN/推論msの一覧表(画像化)

前提: tools/evaluate.py --target suit --save-json ... を先に5条件分実行しておくこと。

使い方:
    python tools/evaluate.py --engine both --target suit --deck-dir data/deck             --save-json results/comparison_suit.json
    python tools/evaluate.py --engine both --target suit --deck-dir data/deck_tilt        --save-json results/stress_test/comparison_suit_tilt.json
    python tools/evaluate.py --engine both --target suit --deck-dir data/deck_tilt_photo  --save-json results/stress_test/comparison_suit_tilt_photo.json
    python tools/evaluate.py --engine both --target suit --deck-dir data/deck_light       --save-json results/stress_test/comparison_suit_light.json
    python tools/evaluate.py --engine both --target suit --deck-dir data/deck_light_photo --save-json results/stress_test/comparison_suit_light_photo.json
    python tools/report_suit_results.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

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
    ("baseline", "通常(E)", RESULTS_DIR / "comparison_suit.json"),
    ("tilt", "傾き(合成)", STRESS_DIR / "comparison_suit_tilt.json"),
    ("tilt_photo", "傾き(実写)", STRESS_DIR / "comparison_suit_tilt_photo.json"),
    ("light", "照明変化(合成)", STRESS_DIR / "comparison_suit_light.json"),
    ("light_photo", "照明変化(実写)", STRESS_DIR / "comparison_suit_light_photo.json"),
]
ENGINE_JA = {
    "classify (OpenCV template matching) [suit]": "古典的CV",
    "classify_yolo [suit]": "YOLOv8",
}


def load_condition_results(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {item["name"]: item for item in payload}


def make_accuracy_chart(all_results: dict[str, dict[str, dict]]) -> None:
    engines = list(ENGINE_JA.keys())
    condition_labels = [label for _key, label, _p in CONDITIONS]
    condition_keys = [key for key, _label, _p in CONDITIONS]

    x = np.arange(len(condition_keys))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    colors = {"classify (OpenCV template matching) [suit]": "#4c9f70", "classify_yolo [suit]": "#e4b73f"}

    for i, engine in enumerate(engines):
        accs = [all_results[key][engine]["accuracy"] * 100 for key in condition_keys]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, accs, width, label=ENGINE_JA[engine], color=colors.get(engine))
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f"{acc:.1f}%", ha="center", va="bottom", fontsize=10)

    ax.set_ylim(0, 110)
    ax.set_ylabel("スート正解率 (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(condition_labels, rotation=10)
    ax.set_title("難条件によるスート正解率の変化(全52枚, 合成 vs 実写)")
    ax.axhline(100, color="gray", linewidth=0.5, linestyle="--")
    ax.legend()
    fig.tight_layout()
    out = STRESS_DIR / "accuracy_chart_suit.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"保存: {out}")


def make_summary_table(all_results: dict[str, dict[str, dict]]) -> None:
    engines = list(ENGINE_JA.keys())
    condition_labels = [label for _key, label, _p in CONDITIONS]
    condition_keys = [key for key, _label, _p in CONDITIONS]

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
    ax.set_title("難条件比較 スート正解率サマリー表(全52枚)", fontsize=13, pad=20)
    fig.tight_layout()
    out = STRESS_DIR / "summary_table_suit.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"保存: {out}")


def main() -> int:
    STRESS_DIR.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, dict[str, dict]] = {}
    for key, _label, json_path in CONDITIONS:
        if not json_path.exists():
            print(f"エラー: 結果ファイルが見つかりません: {json_path}", file=sys.stderr)
            return 1
        all_results[key] = load_condition_results(json_path)

    make_accuracy_chart(all_results)
    make_summary_table(all_results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
