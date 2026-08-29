# card_ten_project

トランプの画像を判別して数字を読み取り、そこから四則演算で目標値(既定10)を
導くプロジェクト。判別部分を**古典的なOpenCV手法**と**YOLOv8**の2通りで実装し、
同じ画像・同じ評価指標で比較検証する。

設計と判断根拠の詳細は [COMPARE.md](COMPARE.md) を参照。

## ファイル構成

| ファイル | 役割 |
|---|---|
| `prepare_deck.py` | 撮影した写真をファイル名=正解ラベルの形式に変換する前処理ツール |
| `common.py` | 定数・データセット読み込み(T/E共通) |
| `classify.py` | **古典的CV手法**: `predict_rank(image_path) -> str \| None` |
| `classify_yolo.py` | **YOLOv8手法**(既存データセット学習済みモデル、ゼロショット): `predict_rank(image_path) -> str \| None` |
| `tools/evaluate.py` | 2手法を同じE・同じ指標で評価する共通スクリプト |
| `make_ten.py` | 読み取った数字から四則演算+括弧で目標値を作る式を全探索するソルバー |
| `demo.py` | Eからカードを引いて判別→`make_ten`で式を出す、実利用寄りのデモ |
| `data/template_src/` | T: テンプレート作成専用画像(セッション1) |
| `data/deck/` | E: 評価+ランダム抽選専用画像(セッション2) |
| `pretrained_models/` | ダウンロード済みのYOLOv8学習済み重み |

## セットアップ

```
pip install -r requirements.txt
```

`classify.py` だけを試すなら `ultralytics` は不要。

## 使い方

### 0. 画像を用意する(まだの場合)

`data/template_src/`(T)と `data/deck/`(E)に、それぞれ別の撮影セッションで
撮った52枚を `prepare_deck.py` で書き出す。詳細は [data/README.md](data/README.md)。

### 1. 比較評価

```
python tools/evaluate.py --engine classical   # 古典CVのみ
python tools/evaluate.py --engine yolo        # YOLOv8のみ
python tools/evaluate.py --engine both --save-json results/comparison.json
```

### 2. 「10を作る」デモ

```
python demo.py --engine classical
python demo.py --engine yolo
python demo.py --cards H10.png C05.png S02.png D08.png --target 15
```

## 現在の状態

- 本番のT・E撮影(52枚×2セッション)完了、`prepare_deck.py` で
  `data/template_src/` と `data/deck/` に書き出し済み。
- 撮影環境は黒クロス背景のため、`classify.py` の背景分離をHSV緑色抽出から
  輝度(Otsu二値化)ベースに変更済み。
- 本番評価が完了([results/comparison.json](results/comparison.json)):
  - 古典CV: accuracy 0.962 (50/52)(誤答2件の原因調査は[COMPARE.md](COMPARE.md) §8.1、可視化画像は`debug_out/`。コードは未変更)
  - YOLOv8: accuracy 1.000 (52/52)
  - 結果の詳細と解釈は [COMPARE.md](COMPARE.md) §8 を参照。
- Webアプリ([webapp/](webapp/README.md))も動作確認済み。
  `uvicorn webapp.backend.main:app --reload` で起動し http://127.0.0.1:8000 を開く。
