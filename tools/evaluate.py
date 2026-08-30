"""古典的CV手法(classify.py)とYOLOv8手法(classify_yolo.py)を、
同じE(data/deck/)・同じ指標で評価する共通スクリプト。

両手法は同じシグネチャの関数を公開する:

    def predict_rank(image_path: Path) -> str | None

この関数を呼ぶだけで評価できるようにすることで、「YOLOに合わせて指標を
後付けした」という疑いが生じないようにしている(このファイルは
classify_yolo.py が存在する前に、classify.py だけを対象として先に
完成させたもの)。

主指標は **ランク正解率 (correct / 52)**。
mAP は使わない。理由:
    - 古典的CV手法(classify.py)には検出(detection)の工程が無く、
      常に「画像1枚 → カード領域1つ」という前提で処理している。
    - 比較を成立させるため、YOLO側も「1画像 → 最高信頼度の1検出 → 1ランク」
      に落とし込み、両手法とも「画像1枚に対してランクを1つ出す分類問題」
      として扱う。この土俵ではmAP(検出の重なり・複数物体を評価する指標)は
      定義できない/意味を持たないため使用しない。
判別不能(古典CV: カード/文字検出失敗、YOLO: 検出0件)は両方とも None を返す
契約になっており、本スクリプトではこれを共通の "UNKNOWN" として扱う
(=不正解の一種として集計しつつ、件数を別途表示する)。

使い方:
    python tools/evaluate.py --engine classical
    python tools/evaluate.py --engine yolo
    python tools/evaluate.py --engine both --save-json results/comparison.json

--target suit でスート(C/S/D/H)の判別精度も同じ枠組みで評価できる
(既定は --target rank で、従来通りランクのみを評価する)。
    python tools/evaluate.py --engine both --target suit --save-json results/comparison_suit.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable

# tools/ の親(プロジェクトルート)を import パスに追加する
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import DEFAULT_DECK_DIR, DEFAULT_TEMPLATE_DIR, RANKS, SUITS, UNKNOWN, CardImage, load_dataset

# 評価対象: "rank" (既定, 従来通り) / "suit" (新規)。
# ラベル一覧・正解ラベルの取り方・使う predict_* 関数がこれで決まる。
TARGET_LABELS: dict[str, tuple[str, ...]] = {"rank": RANKS, "suit": SUITS}
TARGET_TRUE_LABEL: dict[str, Callable[[CardImage], str]] = {
    "rank": lambda item: item.rank,
    "suit": lambda item: item.suit,
}


# --- 評価ロジック ------------------------------------------------------------


class EvalResult:
    def __init__(self, name: str) -> None:
        self.name = name
        self.true_labels: list[str] = []
        self.pred_labels: list[str] = []  # UNKNOWN を含む
        self.elapsed_ms: list[float] = []

    def add(self, true_label: str, pred_label: str | None, elapsed_ms: float) -> None:
        self.true_labels.append(true_label)
        self.pred_labels.append(pred_label if pred_label is not None else UNKNOWN)
        self.elapsed_ms.append(elapsed_ms)

    @property
    def n(self) -> int:
        return len(self.true_labels)

    @property
    def n_correct(self) -> int:
        return sum(t == p for t, p in zip(self.true_labels, self.pred_labels))

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n if self.n else 0.0

    @property
    def n_unknown(self) -> int:
        return sum(p == UNKNOWN for p in self.pred_labels)

    @property
    def mean_inference_ms(self) -> float:
        return sum(self.elapsed_ms) / len(self.elapsed_ms) if self.elapsed_ms else 0.0

    def confusion_matrix(self, labels: list[str]) -> list[list[int]]:
        """行=正解ランク、列=予測ラベル。

        pred_labels には UNKNOWN が含まれうるが、labels(=RANKS)には含めない。
        UNKNOWN への予測は「どのランクの誤検出にもならない(=列を持たない)」
        ものとして数えず、行列には計上しない(件数は n_unknown で別途集計する)。
        """
        index = {label: i for i, label in enumerate(labels)}
        matrix = [[0] * len(labels) for _ in labels]
        for t, p in zip(self.true_labels, self.pred_labels):
            if t in index and p in index:
                matrix[index[t]][index[p]] += 1
        return matrix

    def per_class_prf(self, labels: list[str]) -> dict[str, tuple[float, float, float, int]]:
        cm = self.confusion_matrix(labels)
        out: dict[str, tuple[float, float, float, int]] = {}
        for i, label in enumerate(labels):
            if label == UNKNOWN:
                continue
            tp = cm[i][i]
            # support は「正解ラベルの出現回数」そのもの(UNKNOWN予測で
            # 行列に計上されなかった分も含む)であるべきなので、行列の
            # 行合計ではなく true_labels から直接数える。
            support = sum(t == label for t in self.true_labels)
            pred_total = sum(cm[r][i] for r in range(len(labels)))
            precision = tp / pred_total if pred_total else 0.0
            recall = tp / support if support else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
            out[label] = (precision, recall, f1, support)
        return out

    def macro_f1(self, labels: list[str]) -> float:
        prf = self.per_class_prf(labels)
        f1s = [v[2] for v in prf.values()]
        return sum(f1s) / len(f1s) if f1s else 0.0


def run_eval(
    name: str,
    predict_fn: Callable[[Path], str | None],
    deck_dir: Path,
    true_label_fn: Callable[[CardImage], str] = TARGET_TRUE_LABEL["rank"],
) -> EvalResult:
    items = load_dataset(deck_dir)
    result = EvalResult(name)
    for item in items:
        start = time.perf_counter()
        pred = predict_fn(item.path)
        elapsed_ms = (time.perf_counter() - start) * 1000
        result.add(true_label_fn(item), pred, elapsed_ms)
    return result


def print_report(result: EvalResult, labels: list[str] | None = None) -> None:
    labels = labels if labels is not None else list(RANKS)
    print(f"\n=== {result.name} ===")
    print(f"  ランク正解率 (accuracy)  : {result.accuracy:.3f}  ({result.n_correct}/{result.n})")
    print(f"  マクロF1                : {result.macro_f1(labels):.3f}")
    print(f"  UNKNOWN件数              : {result.n_unknown}/{result.n}")
    print(f"  平均推論時間              : {result.mean_inference_ms:.2f} ms/枚")
    print("  --- クラス別 precision/recall/f1 ---")
    prf = result.per_class_prf(labels)
    for label in labels:
        p, r, f1, support = prf[label]
        print(f"    {label:>3}: precision={p:.2f} recall={r:.2f} f1={f1:.2f} (support={support})")


def print_comparison(results: list[EvalResult], labels: list[str] | None = None) -> None:
    labels = labels if labels is not None else list(RANKS)
    print("\n" + "=" * 70)
    print(f"比較サマリー (E: 全{results[0].n}枚, 主指標=ランク正解率)")
    print("=" * 70)
    print(f"{'手法':<42} {'accuracy':>9} {'macroF1':>8} {'UNKNOWN':>8} {'推論ms/枚':>10}")
    for r in results:
        print(f"{r.name:<42} {r.accuracy:>9.3f} {r.macro_f1(labels):>8.3f} {r.n_unknown:>8d} {r.mean_inference_ms:>10.2f}")


def result_to_dict(result: EvalResult, labels: list[str] | None = None) -> dict:
    labels = labels if labels is not None else list(RANKS)
    return {
        "name": result.name,
        "n": result.n,
        "n_correct": result.n_correct,
        "accuracy": result.accuracy,
        "macro_f1": result.macro_f1(labels),
        "n_unknown": result.n_unknown,
        "mean_inference_ms": result.mean_inference_ms,
        "predictions": [
            {"true": t, "pred": p, "ms": ms}
            for t, p, ms in zip(result.true_labels, result.pred_labels, result.elapsed_ms)
        ],
    }


# --- CLI ---------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine", choices=("classical", "yolo", "both"), default="both")
    parser.add_argument("--target", choices=("rank", "suit"), default="rank", help="評価対象: rank(既定, 従来通り) / suit(新規)")
    parser.add_argument("--deck-dir", type=Path, default=DEFAULT_DECK_DIR, help="E: 評価用画像フォルダ")
    parser.add_argument("--template-dir", type=Path, default=DEFAULT_TEMPLATE_DIR, help="T: 古典CVのテンプレート元フォルダ")
    parser.add_argument("--save-json", type=Path, default=None, help="結果をJSONで保存するパス")
    args = parser.parse_args(argv)

    if not args.deck_dir.is_dir():
        print(f"エラー: E(評価用)フォルダが見つかりません: {args.deck_dir}", file=sys.stderr)
        return 1

    labels = list(TARGET_LABELS[args.target])
    true_label_fn = TARGET_TRUE_LABEL[args.target]
    name_suffix = "" if args.target == "rank" else " [suit]"

    results: list[EvalResult] = []

    if args.engine in ("classical", "both"):
        import classify
        classify.configure(args.template_dir)
        predict_fn = classify.predict_rank if args.target == "rank" else classify.predict_suit
        print(f"[古典的CV手法] 評価中... (対象={args.target}, テンプレート元 T = {args.template_dir})")
        result = run_eval(classify.__name__ + " (OpenCV template matching)" + name_suffix, predict_fn, args.deck_dir, true_label_fn)
        print_report(result, labels)
        results.append(result)

    if args.engine in ("yolo", "both"):
        try:
            import classify_yolo
        except ImportError as e:
            print(f"\n[YOLOv8手法] スキップ: {e}", file=sys.stderr)
        else:
            predict_fn = classify_yolo.predict_rank if args.target == "rank" else classify_yolo.predict_suit
            print(f"\n[YOLOv8手法] 評価中... (対象={args.target}, 既存データセット学習済みモデル, ゼロショット)")
            result = run_eval(classify_yolo.__name__ + name_suffix, predict_fn, args.deck_dir, true_label_fn)
            print_report(result, labels)
            results.append(result)

    if len(results) > 1:
        print_comparison(results, labels)

    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        payload = [result_to_dict(r, labels) for r in results]
        args.save_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n結果を保存しました: {args.save_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
