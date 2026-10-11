"""FastAPI backend: leaderboards and a live Stuff+ grader."""
from __future__ import annotations

import json
import math
import os
import uuid
from datetime import datetime, timezone
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
CUSTOM = Path(__file__).resolve().parent.parent / "custom_pitchers.json"


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


class CustomPitcher(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    hand: Literal["R", "L"] = "R"
    pitches: list[Pitch] = Field(min_length=1, max_length=12)
    notes: str = Field("", max_length=500)


def create_app(bundle_path: Path, custom_path: Path = CUSTOM) -> FastAPI:
    bundle = joblib.load(bundle_path)
    pitchers, pitch_types = bundle["pitchers"], bundle["pitch_types"]
    app = FastAPI(title="Stuff+ leaderboards")

    @app.get("/api/meta")
    def meta():
        counts = pitch_types.groupby("pitch_type")["n"].sum().sort_values(ascending=False)
        return {
            "variant": bundle["variant"], "features": bundle["features"],
            "label": bundle.get("label", bundle["variant"]), "spread": bundle.get("spread"), "frozen": bundle.get("frozen", False),
            "season_notes": {str(k): v for k, v in bundle.get("season_notes", {}).items()},
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

    # Saved custom pitchers live in one JSON file next to the repo's code, so
    # they survive restarts and rebuilds (gitignored; back it up if it matters).
    def read_custom():
        try:
            return json.loads(Path(custom_path).read_text())
        except FileNotFoundError:
            return []

    def write_custom(rows):
        tmp = Path(f"{custom_path}.tmp")
        tmp.write_text(json.dumps(rows, indent=1))
        os.replace(tmp, custom_path)

    def graded(row, season):
        return {**row, "result": grade_arsenal(bundle, row["hand"], row["pitches"], season)}

    @app.get("/api/custom")
    def list_custom(season: Optional[int] = None):
        if season is not None and season not in bundle["seasons"]:
            raise HTTPException(404, f"no season {season}")
        return [graded(r, season) for r in read_custom()]

    @app.post("/api/custom")
    def create_custom(p: CustomPitcher):
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        row = {"id": uuid.uuid4().hex[:12], **p.model_dump(), "created": now, "updated": now}
        write_custom(read_custom() + [row])
        return graded(row, None)

    @app.put("/api/custom/{cid}")
    def update_custom(cid: str, p: CustomPitcher):
        rows = read_custom()
        for i, r in enumerate(rows):
            if r["id"] == cid:
                rows[i] = {**r, **p.model_dump(),
                           "updated": datetime.now(timezone.utc).isoformat(timespec="seconds")}
                write_custom(rows)
                return graded(rows[i], None)
        raise HTTPException(404, "unknown custom pitcher")

    @app.delete("/api/custom/{cid}")
    def delete_custom(cid: str):
        rows = read_custom()
        keep = [r for r in rows if r["id"] != cid]
        if len(keep) == len(rows):
            raise HTTPException(404, "unknown custom pitcher")
        write_custom(keep)
        return {"deleted": cid}

    # The page and its script must always match the running server, so make
    # browsers revalidate them instead of reusing a cached copy after an update.
    @app.middleware("http")
    async def revalidate(request, call_next):
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
