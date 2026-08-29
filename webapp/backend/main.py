"""10を作るカードゲーム — FastAPIバックエンド。

構成は https://github.com/0-s0g0/grayscale-app/blob/main/setup.md の
backend/frontend分離 + FastAPI + vanilla JS フロントエンドという構成を
土台にした(バックエンドがAPIを提供し、フロントエンドはFetch APIで叩くだけの
素朴な構成)。

エンドポイント:
    GET /api/draw?engine=yolo|classical&count=4&target=10&seed=...&condition=normal|tilt|light
        指定した撮影条件のフォルダからcount枚引き、選んだエンジンで判別し、
        targetを作る式を返す。condition省略時は元のE(通常条件)のまま
        (=既存の挙動を変えない)。
    GET /images/normal/{filename}  … data/deck/       (E, 通常条件)
    GET /images/tilt/{filename}    … data/deck_tilt/  (傾き, tools/make_stress_tests.py生成)
    GET /images/light/{filename}   … data/deck_light/ (照明変化, 同上)
    GET /
        フロントエンド(webapp/frontend/)を配信する。

起動:
    cd card_ten_project
    .venv\\Scripts\\activate
    uvicorn webapp.backend.main:app --reload
    ブラウザで http://127.0.0.1:8000 を開く
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

_BACKEND_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _BACKEND_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_BACKEND_DIR))  # uvicorn が "webapp.backend.main" として
                                        # importするとbackend/自体はパスに乗らないため

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from common import DEFAULT_DECK_DIR  # noqa: E402
import game  # noqa: E402
from game import CONDITION_DIRS  # noqa: E402

app = FastAPI(title="10を作るカードゲーム")

# フロントエンドをファイルから直接開いた場合(file://)でも叩けるように許可しておく
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/draw")
def api_draw(
    engine: str = Query("yolo", pattern="^(yolo|classical)$"),
    count: int = Query(4, ge=2, le=7),
    target: int = Query(10, ge=1, le=999),
    seed: Optional[int] = Query(None),
    condition: str = Query("normal", pattern="^(normal|tilt|light)$"),
):
    deck_dir = CONDITION_DIRS.get(condition, DEFAULT_DECK_DIR)
    if not deck_dir.is_dir():
        raise HTTPException(
            status_code=500,
            detail=f"画像フォルダが見つかりません({condition}): {deck_dir}",
        )
    return game.draw_and_solve(engine=engine, count=count, target=target, seed=seed, condition=condition)


# --- 静的ファイル配信 --------------------------------------------------------
# 特定パスのAPIルートを先に定義してから、最後に汎用マウントを置く
# (Starletteはルート登録順に一致判定するため、この順序が重要)。

for _condition, _dir in CONDITION_DIRS.items():
    if _dir.is_dir():
        app.mount(f"/images/{_condition}", StaticFiles(directory=str(_dir)), name=f"images_{_condition}")

_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend")
