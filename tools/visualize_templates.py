"""T(data/template_src/)から作られた13ランク分のテンプレート画像を
一覧にまとめ、`results/templates_overview.png` として保存するだけの
スクリプト(Webアプリには反映しない)。

使い方:
    python tools/visualize_templates.py
    python tools/visualize_templates.py --template-dir data/template_src
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import classify
from common import DEFAULT_TEMPLATE_DIR, PROJECT_ROOT, RANKS

OUT_PATH = PROJECT_ROOT / "results" / "templates_overview.png"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--template-dir", type=Path, default=DEFAULT_TEMPLATE_DIR, help="T: テンプレート元フォルダ")
    parser.add_argument("--out", type=Path, default=OUT_PATH, help="出力先パス")
    args = parser.parse_args(argv)

    classify.configure(args.template_dir)
    templates = classify._get_templates()

    labels = [r for r in RANKS if r in templates]
    missing = [r for r in RANKS if r not in templates]
    if missing:
        print(f"[警告] テンプレートが無いランク: {missing}")

    cols = 7
    rows = -(-len(labels) // cols)  # 切り上げ
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 2.0))
    axes = axes.reshape(rows, cols)

    for i in range(rows * cols):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if i < len(labels):
            rank = labels[i]
            ax.imshow(templates[rank], cmap="gray", vmin=0, vmax=255)
            ax.set_title(rank, fontsize=14)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("#cccccc")

    fig.suptitle(
        f"テンプレート一覧 ({len(labels)}ランク, 元: {args.template_dir.name})",
        fontsize=16,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"保存: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
