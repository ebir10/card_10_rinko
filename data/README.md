# data/

| フォルダ | セット名 | 用途 | 撮影 |
|---|---|---|---|
| `template_src/` | T | `classify.py` のテンプレート作成専用。評価には使わない | セッション1 |
| `deck/` | E | 両手法(`classify.py` / `classify_yolo.py`)の評価専用、かつ `demo.py` のランダム抽選元 | セッション2 |
| `_smoketest_only/` | (仮データ) | コード動作確認のみに使った、他プロジェクト(`crin`)からの借用データ。**最終評価には使わない** | — |

T と E は別の撮影セッションであること(理想は別日)。詳しい理由は
[../COMPARE.md](../COMPARE.md) §1 を参照。

撮影後は `prepare_deck.py` で書き出す:

```
python prepare_deck.py --src <撮影した写真のフォルダ> --dst data/template_src --apply
python prepare_deck.py --src <撮影した写真のフォルダ> --dst data/deck --apply
```

ファイル名は `SUIT+RANK.png` (例: `H10.png` = ハートの10)。
