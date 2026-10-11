"""FastAPI backend: leaderboards and a live Stuff+ grader."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Literal, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from webapp.grading import grade_arsenal

STATIC = Path(__file__).parent / "static"


def _records(df: pd.DataFrame):
    """DataFrame -> JSON-safe list of dicts (NaN -> None)."""
    out = df.to_dict("records")
    for r in out:
        for k, v in r.items():
            if isinstance(v, (float, np.floating)) and not math.isfinite(v):
                r[k] = None
            elif isinstance(v, np.generic):
                r[k] = v.item()
    return out


class Pitch(BaseModel):
    pitch_type: str = Field(examples=["FF"])
    usage: float = Field(50, ge=0, le=100, description="% of pitches")
    velo: float = Field(ge=50, le=110, description="release speed, mph")
    spin: float = Field(ge=0, le=4000, description="spin rate, rpm")
    ext: float = Field(6.3, ge=3, le=9, description="extension, ft")
    rel_side: float = Field(1.8, ge=-4, le=5, description="release side, ft, + = arm side")
    rel_height: float = Field(5.8, ge=2, le=8, description="release height, ft")
    ivb: float = Field(ge=-30, le=30, description="induced vertical break, in")
    hb: float = Field(ge=-30, le=30, description="horizontal break, in, + = arm side")
    spin_axis: Optional[float] = Field(None, ge=0, le=360)
    arm_angle: Optional[float] = Field(None, ge=-90, le=90)


class GradeRequest(BaseModel):
    hand: Literal["R", "L"] = "R"
    pitches: list[Pitch] = Field(min_length=1, max_length=12)
    season: Optional[int] = None


def create_app(bundle_path: Path) -> FastAPI:
    bundle = joblib.load(bundle_path)
    pitchers, pitch_types = bundle["pitchers"], bundle["pitch_types"]
    app = FastAPI(title="Stuff+ leaderboards")

    @app.get("/api/meta")
    def meta():
        counts = pitch_types.groupby("pitch_type")["n"].sum().sort_values(ascending=False)
        return {
            "variant": bundle["variant"], "features": bundle["features"],
            "train_years": bundle["train_years"], "seasons": bundle["seasons"],
            "latest_season": bundle["latest_season"], "synthetic": bundle["synthetic"],
            "built_at": bundle["built_at"], "versions": bundle["versions"],
            "pitch_types": counts.index.tolist(),
            "has_roles": "role" in pitchers.columns,
            "grades": {str(k): v for k, v in bundle["grades"].items()},
        }

    @app.get("/api/leaderboard/{season}")
    def leaderboard(season: int):
        if season not in bundle["seasons"]:
            raise HTTPException(404, f"no season {season}")
        return {"pitchers": _records(pitchers[pitchers["year"] == season]),
                "pitch_types": _records(pitch_types[pitch_types["year"] == season])}

    @app.get("/api/pitcher/{pitcher_id}")
    def pitcher(pitcher_id: int):
        p = pitchers[pitchers["pitcher"] == pitcher_id]
        if p.empty:
            raise HTTPException(404, "unknown pitcher")
        return {"seasons": _records(p.sort_values("year")),
                "pitch_types": _records(pitch_types[pitch_types["pitcher"] == pitcher_id]
                                        .sort_values(["year", "n"], ascending=[True, False]))}

    @app.post("/api/grade")
    def grade(req: GradeRequest):
        if req.season is not None and req.season not in bundle["seasons"]:
            raise HTTPException(404, f"no season {req.season}")
        return grade_arsenal(bundle, req.hand, [p.model_dump() for p in req.pitches], req.season)

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
