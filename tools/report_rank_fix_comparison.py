"""ランク文字/スートマークの行分離バグ修正(classify.py の ROW_GAP_TOL・
MIN_GLYPH_WIDTH変更)の前後で、ランク正解率がどう変わったかをまとめる。

出力(results/rank_fix/):
    before_after_accuracy_chart.png … 古典的CVのみの正解率比較グラフ
    before_after_summary_table.png  … 古典的CV・YOLOv8 両方の
        修正前/修正後(accuracy・macroF1・UNKNOWN)を並べた一覧表

前提: results/rank_fix/before/ と results/rank_fix/after/ に、それぞれ
修正前・修正後の comparison*.json 一式が保存されていること
(このリポジトリでは過去のgitコミットから復元済み)。

使い方:
    python tools/report_rank_fix_comparison.py
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

RANK_FIX_DIR = PROJECT_ROOT / "results" / "rank_fix"

CONDITIONS = [
    ("baseline", "通常(E)", "comparison.json"),
    ("tilt", "傾き(合成)", "stress_test/comparison_tilt.json"),
    ("tilt_photo", "傾き(実写)", "stress_test/comparison_tilt_photo.json"),
    ("light", "照明変化(合成)", "stress_test/comparison_light.json"),
    ("light_photo", "照明変化(実写)", "stress_test/comparison_light_photo.json"),
]
CLASSICAL_NAME = "classify (OpenCV template matching)"
YOLO_NAME = "classify_yolo"
ENGINES = [(CLASSICAL_NAME, "古典的CV"), (YOLO_NAME, "YOLOv8")]


def load_result(snapshot_dir: Path, rel_path: str, name: str) -> dict:
    payload = json.loads((snapshot_dir / rel_path).read_text(encoding="utf-8"))
    for item in payload:
        if item["name"] == name:
            return item
    raise KeyError(f"{name} が見つかりません: {snapshot_dir / rel_path}")


def load_classical_accuracy(snapshot_dir: Path, rel_path: str) -> float:
    return load_result(snapshot_dir, rel_path, CLASSICAL_NAME)["accuracy"] * 100


def make_summary_table() -> None:
    rows = []
    for _key, label, rel_path in CONDITIONS:
        for name, engine_ja in ENGINES:
            b = load_result(RANK_FIX_DIR / "before", rel_path, name)
            a = load_result(RANK_FIX_DIR / "after", rel_path, name)
            rows.append([
                label, engine_ja,
                f"{b['accuracy']*100:.1f}%", f"{a['accuracy']*100:.1f}%",
                f"{b['macro_f1']:.3f}", f"{a['macro_f1']:.3f}",
                f"{b['n_unknown']}", f"{a['n_unknown']}",
            ])

    col_labels = ["条件", "手法", "accuracy(前)", "accuracy(後)", "macroF1(前)", "macroF1(後)", "UNKNOWN(前)", "UNKNOWN(後)"]
    fig, ax = plt.subplots(figsize=(11.5, 0.5 * len(rows) + 1))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    for c in range(len(col_labels)):
        table[0, c].set_facecolor("#2c4a35")
        table[0, c].set_text_props(color="white", weight="bold")
    # accuracy/macroF1が前より改善したセルを薄緑、UNKNOWNが前より増えたセルを薄橙で強調
    # (UNKNOWNは増えても正答率が上がっているケースがあるため、単純な「変化あり」ではなく
    # 良化/悪化の向きで色を分ける)
    for r, row in enumerate(rows, start=1):
        b_acc, a_acc = float(row[2].rstrip("%")), float(row[3].rstrip("%"))
        if a_acc > b_acc:
            table[r, 3].set_facecolor("#dff0d8")
        b_f1, a_f1 = float(row[4]), float(row[5])
        if a_f1 > b_f1:
            table[r, 5].set_facecolor("#dff0d8")
        b_unk, a_unk = int(row[6]), int(row[7])
        if a_unk > b_unk:
            table[r, 7].set_facecolor("#fbe5d6")
        elif a_unk < b_unk:
            table[r, 7].set_facecolor("#dff0d8")
    ax.set_title("行分離バグ修正の前後比較サマリー表(全52枚)", fontsize=13, pad=20)
    fig.tight_layout()
    out = RANK_FIX_DIR / "before_after_summary_table.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"保存: {out}")


def main() -> int:
    condition_labels = [label for _key, label, _p in CONDITIONS]
    before = [load_classical_accuracy(RANK_FIX_DIR / "before", p) for _k, _l, p in CONDITIONS]
    after = [load_classical_accuracy(RANK_FIX_DIR / "after", p) for _k, _l, p in CONDITIONS]

    x = np.arange(len(CONDITIONS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    bars_before = ax.bar(x - width / 2, before, width, label="修正前(行融合バグあり)", color="#c0745a")
    bars_after = ax.bar(x + width / 2, after, width, label="修正後", color="#4c9f70")
    for bars in (bars_before, bars_after):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                     f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=10)

    ax.set_ylim(0, 110)
    ax.set_ylabel("ランク正解率 (%, 古典的CVのみ)")
    ax.set_xticks(x)
    ax.set_xticklabels(condition_labels, rotation=10)
    ax.set_title("行分離バグ修正の前後比較(古典的CV, 全52枚)")
    ax.axhline(100, color="gray", linewidth=0.5, linestyle="--")
    ax.legend()
    fig.tight_layout()
    out = RANK_FIX_DIR / "before_after_accuracy_chart.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"保存: {out}")

    make_summary_table()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
