"""ランク文字/スートマークの行分離バグ修正(classify.py の ROW_GAP_TOL・
MIN_GLYPH_WIDTH変更)の前後で、古典的CVのランク正解率がどう変わったかを
1枚のグラフにまとめる。

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


def load_classical_accuracy(snapshot_dir: Path, rel_path: str) -> float:
    payload = json.loads((snapshot_dir / rel_path).read_text(encoding="utf-8"))
    for item in payload:
        if item["name"] == CLASSICAL_NAME:
            return item["accuracy"] * 100
    raise KeyError(f"{CLASSICAL_NAME} が見つかりません: {snapshot_dir / rel_path}")


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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
