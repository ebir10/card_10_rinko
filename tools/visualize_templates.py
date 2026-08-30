"""T(data/template_src/)から作られた13ランク分・4スート分のテンプレート
画像を一覧にまとめ、`results/templates_overview.png` /
`results/templates_overview_suit.png` として保存するだけのスクリプト
(Webアプリには反映しない)。

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
from common import DEFAULT_TEMPLATE_DIR, PROJECT_ROOT, RANKS, SUITS, SUIT_MARK

OUT_PATH = PROJECT_ROOT / "results" / "templates_overview.png"
OUT_PATH_SUIT = PROJECT_ROOT / "results" / "templates_overview_suit.png"


def save_grid(templates: dict, labels: list[str], cols: int, title: str, out: Path, label_of=lambda x: x) -> None:
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
            label = labels[i]
            ax.imshow(templates[label], cmap="gray", vmin=0, vmax=255)
            ax.set_title(label_of(label), fontsize=14)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("#cccccc")

    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"保存: {out}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--template-dir", type=Path, default=DEFAULT_TEMPLATE_DIR, help="T: テンプレート元フォルダ")
    parser.add_argument("--out", type=Path, default=OUT_PATH, help="ランク一覧の出力先パス")
    parser.add_argument("--out-suit", type=Path, default=OUT_PATH_SUIT, help="スート一覧の出力先パス")
    args = parser.parse_args(argv)

    classify.configure(args.template_dir)

    templates = classify._get_templates()
    rank_labels = [r for r in RANKS if r in templates]
    missing = [r for r in RANKS if r not in templates]
    if missing:
        print(f"[警告] テンプレートが無いランク: {missing}")
    save_grid(
        templates, rank_labels, cols=7,
        title=f"テンプレート一覧 ({len(rank_labels)}ランク, 元: {args.template_dir.name})",
        out=args.out,
    )

    suit_templates = classify._get_suit_templates()
    suit_labels = [s for s in SUITS if s in suit_templates]
    missing_suits = [s for s in SUITS if s not in suit_templates]
    if missing_suits:
        print(f"[警告] テンプレートが無いスート: {missing_suits}")
    save_grid(
        suit_templates, suit_labels, cols=4,
        title=f"スートテンプレート一覧 ({len(suit_labels)}スート, 元: {args.template_dir.name})",
        out=args.out_suit,
        label_of=lambda s: f"{s} {SUIT_MARK.get(s, '')}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
