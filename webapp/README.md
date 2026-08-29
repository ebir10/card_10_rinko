# 10を作るカードゲーム — Webアプリ

E(`data/deck/`)からランダムに数枚のカードを引き、選んだエンジン
(YOLOv8 / 古典的CV)でランクを判別し、読み取った数字から四則演算で
目標値(既定10)を作る式を探すWebアプリ。

構成は [grayscale-app](https://github.com/0-s0g0/grayscale-app/blob/main/setup.md)
と同じ「FastAPIバックエンド + vanilla JSフロントエンド」を土台にしている。

```
webapp/
├── backend/
│   ├── main.py            FastAPIサーバー本体(APIエンドポイント + 静的配信)
│   ├── game.py            カードを引いて判別し、10を作る式を探すロジック
│   └── requirements.txt   webapp用の追加依存(fastapi, uvicorn)
└── frontend/
    ├── index.html
    ├── style.css
    └── script.js          Fetch APIでバックエンドを叩くだけの素朴なJS
```

## セットアップ・起動

プロジェクトルート(`card_ten_project/`)の既存venvをそのまま使う
(`classify.py` / `classify_yolo.py` の依存が既に入っているため)。

```
cd card_ten_project
.venv\Scripts\activate
pip install -r webapp\backend\requirements.txt
uvicorn webapp.backend.main:app --reload
```

ブラウザで **http://127.0.0.1:8000** を開く。

## 仕組み

- `GET /api/draw?engine=yolo|classical&count=4&target=10&condition=normal|tilt|light`
  — 指定した撮影条件のフォルダからcount枚引き、`classify.predict_rank` /
  `classify_yolo.predict_rank` (プロジェクトルートのモジュールをそのまま
  importして使う)でランクを判別し、`make_ten.solve()` でtargetを作る式を
  探してJSONで返す。`condition`省略時は`normal`(通常のE)。
- `GET /images/normal/<filename>` — `data/deck/`(E, 通常条件)
  `GET /images/tilt/<filename>` — `data/deck_tilt/`(傾き)
  `GET /images/light/<filename>` — `data/deck_light/`(照明変化)
  いずれも`tools/make_stress_tests.py`が生成した、正解ラベル・ファイル名は
  Eと共通のセット。`StaticFiles`マウントでそのまま配信する。
- `GET /` — `frontend/` を配信する(同一オリジンなのでCORSの心配が無いが、
  念のためCORSミドルウェアも許可済み)。

画面上部の「撮影条件」ドロップダウンで通常/傾き/照明変化を切り替えられる。
どの条件でも同じ52枚(同じランク構成)から引くので、エンジン・条件を変えて
何度も引き比べることで、[COMPARE.md](../COMPARE.md) §9 の難条件比較を
インタラクティブに追体験できる。

計算に使う数字は**実際に画像処理/YOLOで読み取った値**であり、ファイル名から
取れる正解ラベルではない(このプロジェクト全体のテーマ「画像処理で読み取った
数字から10を導く」に沿っている)。判別できなかったカード(UNKNOWN)は
計算から除外され、その旨が画面に表示される。
