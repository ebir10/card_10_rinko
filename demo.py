"""E(data/deck/)からカードを何枚か引き、指定したエンジンでランクを判別して
読み取った数字から「10を作る」式を探すデモ。

classify.py / classify_yolo.py どちらも predict_rank(image_path) -> str | None
という同じ契約なので、エンジンの切り替えは importするモジュールを変えるだけ。

使い方:
    python demo.py                                   # 古典CVで4枚引いて10を作る
    python demo.py --engine yolo
    python demo.py --cards H10.png C05.png S02.png D08.png --target 15
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from common import DEFAULT_DECK_DIR, RANK_VALUE, UNKNOWN, load_dataset
import make_ten


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--deck-dir", type=Path, default=DEFAULT_DECK_DIR, help="E: 評価/抽選用画像フォルダ")
    parser.add_argument("--engine", choices=("classical", "yolo"), default="classical")
    parser.add_argument("--count", type=int, default=4, help="引くカードの枚数(既定4)")
    parser.add_argument("--target", type=int, default=10, help="目標の数(既定10)")
    parser.add_argument("--cards", nargs="+", help="ランダム抽選の代わりに使うファイル名")
    parser.add_argument("--seed", type=int, default=None, help="乱数シード(再現用)")
    args = parser.parse_args(argv)

    if not args.deck_dir.is_dir():
        print(f"エラー: E(評価用)フォルダが見つかりません: {args.deck_dir}", file=sys.stderr)
        return 1

    if args.engine == "classical":
        import classify as engine
    else:
        import classify_yolo as engine

    items = load_dataset(args.deck_dir)
    by_name = {it.path.name: it for it in items}

    rng = random.Random(args.seed)
    if args.cards:
        missing = [name for name in args.cards if name not in by_name]
        if missing:
            print(f"エラー: 画像が見つかりません: {missing}", file=sys.stderr)
            return 1
        drawn = [by_name[name] for name in args.cards]
    else:
        drawn = rng.sample(items, args.count)

    print(f"認識エンジン: {args.engine} ({engine.__name__})")
    print(f"\n--- {len(drawn)} 枚のカードを判別します ---")
    values = []
    correct = 0
    for item in drawn:
        pred = engine.predict_rank(item.path)
        pred_label = pred if pred is not None else UNKNOWN
        ok = pred == item.rank
        correct += ok
        print(f"  {item.path.name:>8}  ->  判別結果 {pred_label:>7}  正解 {item.rank:>3}  [{'OK' if ok else 'NG'}]")
        if pred is None:
            print(f"    -> 判別できなかったため、この後の計算からは除外します")
            continue
        values.append(RANK_VALUE[pred])

    print(f"\n判別精度: {correct}/{len(drawn)}")
    print(f"読み取った数字: {values}")

    if len(values) < len(drawn):
        print("(UNKNOWNが含まれるため、実際に計算に使う枚数が減っています)")

    expr = make_ten.solve(values, args.target) if values else None
    if expr is None:
        print(f"\n{args.target} を作る式は見つかりませんでした。")
        return 0
    print(f"\n{expr} = {args.target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
