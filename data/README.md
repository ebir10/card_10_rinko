# data/

| フォルダ | セット名 | 用途 | 撮影 |
|---|---|---|---|
| `template_src/` | T | `classify.py` のテンプレート作成専用。評価には使わない | セッション1 |
| `deck/` | E | 両手法(`classify.py` / `classify_yolo.py`)の評価専用、かつ `demo.py` のランダム抽選元(通常条件) | セッション2 |
| `deck_tilt/` | E-傾き(合成) | `tools/make_stress_tests.py` がEを画像処理で加工して生成。新規撮影ではない | — |
| `deck_light/` | E-照明変化(合成) | 同上 | — |
| `photos_tilt_raw/` → `deck_tilt_photo/` | E-傾き(実写) | Eと同じ52枚の構成を、意図的に傾けて撮り直したもの | セッション3 |
| `photos_light_raw/` → `deck_light_photo/` | E-照明変化(実写) | Eと同じ52枚の構成を、明るめの照明で撮り直したもの(52枚全部同じ照明設定) | セッション4 |
| `_smoketest_only/` | (仮データ) | コード動作確認のみに使った、他プロジェクト(`crin`)からの借用データ。**最終評価には使わない** | — |

T と E は別の撮影セッションであること(理想は別日)。詳しい理由は
[../COMPARE.md](../COMPARE.md) §1 を参照。

**傾き/照明変化の実写(セッション3・4)は、Tはそのまま(通常条件のテンプレート)
を使い、Eと同じ52枚構成をその条件下で撮り直したもの。** これにより
「通常条件で作ったテンプレート/モデルが、難条件でどこまで通用するか」
という§9の評価設計を、合成データではなく実写データで検証できる。
合成版(`deck_tilt/`, `deck_light/`)は削除せず、比較の参考として残す。

撮影後は `prepare_deck.py` で書き出す:

```
python prepare_deck.py --src <撮影した写真のフォルダ> --dst data/template_src --apply
python prepare_deck.py --src <撮影した写真のフォルダ> --dst data/deck --apply

# 傾き・照明変化の実写(セッション3・4)
python prepare_deck.py --src data/photos_tilt_raw  --dst data/deck_tilt_photo  --apply
python prepare_deck.py --src data/photos_light_raw --dst data/deck_light_photo --apply
```

ファイル名は `SUIT+RANK.png` (例: `H10.png` = ハートの10)。並び順(数札40→
絵札12、♣♠♦♥の順)はT・Eと同じにすること。
