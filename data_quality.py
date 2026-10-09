"""
Data-quality report for the cached Statcast seasons. Run before any modeling:

    python data_quality.py                # writes reports/data_quality.md + .json

Covers rows/games per season, null rates per model input, pitch-type mix,
duplicate pitches, run-value target coverage, and a physics check of the
re-aiming code against Statcast's own trajectory fit on real rows.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import physics
from features import add_target, plain_dtypes
from fetch_data import KEY
from run_ablation import env_info

HERE = Path(__file__).parent
INPUTS = ["pitch_type", "release_speed", "release_spin_rate", "spin_axis", "release_extension",
          "arm_angle", "release_pos_x", "release_pos_y", "release_pos_z",
          "vx0", "vy0", "vz0", "ax", "ay", "az", "pfx_x", "pfx_z", "plate_x", "plate_z",
          "p_throws", "stand", "balls", "strikes", "description",
          "estimated_woba_using_speedangle", "delta_run_exp"]
PITCH_TYPES = ["FF", "SI", "FC", "SL", "ST", "SV", "CU", "KC", "CH", "FS", "FO", "KN", "EP",
               "CS", "SC", "FA", "PO"]
EXCLUDED = ["PO", "IN", "EP", "KN", "FA", "CS", "SC"]     # dropped by features.load


def md_table(df, fmt="{:.2f}", index_name=""):
    df = df.copy()
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join([index_name] + cols) + " |",
             "|" + "|".join(["---"] + ["---:"] * len(cols)) + "|"]
    for idx, row in df.iterrows():
        cells = [fmt.format(v) if isinstance(v, (float, np.floating)) and not pd.isna(v)
                 else ("–" if pd.isna(v) else f"{v:,}" if isinstance(v, (int, np.integer)) else str(v))
                 for v in row]
        lines.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join(lines)


def raw_duplicates(data_dir, year):
    """Duplicates on the pitch key in the chunk files, before fetch_data dedups."""
    cdir = data_dir / "chunks" / str(year)
    if not cdir.exists():
        return None
    k = pd.concat([pd.read_parquet(f, columns=KEY + ["game_type"])
                   for f in sorted(cdir.glob("*.parquet")) if pd.read_parquet(f).shape[0]])
    k = k[k["game_type"] == "R"]
    return int(k.duplicated(KEY).sum())


def physics_check(df):
    """Compare physics.py against Statcast's own fit on real pitches."""
    need = ["vx0", "vy0", "vz0", "ax", "ay", "az", "release_extension", "release_pos_x",
            "release_pos_y", "release_pos_z", "plate_x", "plate_z"]
    d = df.dropna(subset=need)
    c = {k: d[k].to_numpy(dtype=float) for k in need}
    traj = (c["vx0"], c["vy0"], c["vz0"], c["ax"], c["ay"], c["az"])
    out = {"n": len(d)}

    def q(name, x):
        x = np.abs(x[np.isfinite(x)])
        out[name] = {"median": float(np.median(x)), "p99": float(np.percentile(x, 99)),
                     "max": float(x.max())}

    # 1. release plane: is release_pos_y == 60.5 - extension?
    q("release_pos_y - (60.5 - extension) [ft]", c["release_pos_y"] - (60.5 - c["release_extension"]))
    # 2. fit rewound from the plate crossing to the release plane vs release_pos_x/z
    for label, y_r in [("60.5 - extension", 60.5 - c["release_extension"]),
                       ("release_pos_y", c["release_pos_y"])]:
        x, z = physics.fit_position_at(y_r, c["plate_x"], c["plate_z"], *traj)
        q(f"fit x at {label} - release_pos_x [ft]", x - c["release_pos_x"])
        q(f"fit z at {label} - release_pos_z [ft]", z - c["release_pos_z"])
    # 3. re-aim at the pitch's own plate location vs its actual VAA/HAA
    vaa0, haa0 = physics.actual_angles(*traj)
    for label, kw in [("release_pos_x/z (as used for features)",
                       dict(release_x=c["release_pos_x"], release_z=c["release_pos_z"])),
                      ("fit-consistent release point", None)]:
        if kw is None:
            xr, zr = physics.fit_position_at(60.5 - c["release_extension"], c["plate_x"],
                                             c["plate_z"], *traj)
            kw = dict(release_x=xr, release_z=zr)
        x_r, z_r, y_r, vy_r = physics.release_state(*traj, c["release_extension"], **kw)
        vaa, haa = physics.angles_at_target(x_r, z_r, y_r, vy_r, c["ax"], c["ay"], c["az"],
                                            c["plate_x"], c["plate_z"])
        q(f"re-aimed VAA - actual VAA, {label} [deg]", vaa - vaa0)
        q(f"re-aimed HAA - actual HAA, {label} [deg]", haa - haa0)
    # 4. plausibility of the location-neutral features themselves
    g = physics.grid_angles(d[need + ["p_throws"]].copy())
    out["grid_nonfinite_rows"] = int((~np.isfinite(g[physics.VAA_GRID_COLS + physics.HAA_GRID_COLS]
                                                    .to_numpy())).any(axis=1).sum())
    out["vaa_ln_range"] = [float(g["vaa_ln"].quantile(x)) for x in (0.001, 0.5, 0.999)]
    out["haa_ln_range"] = [float(g["haa_ln"].quantile(x)) for x in (0.001, 0.5, 0.999)]
    ff = d["pitch_type"].to_numpy() == "FF"
    out["ff_vaa_ln_mean"] = float(g["vaa_ln"][ff].mean())
    out["ff_haa_ln_mean_by_hand"] = {h: float(g["haa_ln"][ff & (d["p_throws"].to_numpy() == h)].mean())
                                     for h in ("L", "R")}
    t_r = physics._time_to_y(physics.Y50, c["vy0"], c["ay"], y_r)
    speed = np.sqrt(vy_r ** 2 + (c["vx0"] + c["ax"] * t_r) ** 2 + (c["vz0"] + c["az"] * t_r) ** 2)
    q("fit speed at release - release_speed [mph]",
      speed * 3600 / 5280 - d["release_speed"].to_numpy(float))
    out["implausible_rows"] = {
        "release_speed outside 40-106 mph": int(((d["release_speed"] < 40) | (d["release_speed"] > 106)).sum()),
        "extension outside 3-9 ft": int(((d["release_extension"] < 3) | (d["release_extension"] > 9)).sum()),
        "release_pos_z outside 0.5-8 ft": int(((d["release_pos_z"] < 0.5) | (d["release_pos_z"] > 8)).sum()),
        "|vaa_ln| > 15 deg": int((g["vaa_ln"].abs() > 15).sum()),
        "ay <= 0": int((d["ay"] <= 0).sum()),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="*", type=int, default=[2022, 2023, 2024, 2025, 2026])
    args = ap.parse_args()
    data_dir = HERE / "data"

    seasons, nulls, mix, phys, dtypes = {}, {}, {}, {}, {}
    for y in args.years:
        df = pd.read_parquet(data_dir / f"statcast_{y}.parquet")
        dtypes[y] = {c: str(t) for c, t in df.dtypes.items()}
        df = plain_dtypes(df)
        gd = pd.to_datetime(df["game_date"])
        df = add_target(df)
        typed = df[df["pitch_type"].notna() & ~df["pitch_type"].isin(EXCLUDED)]
        per_game = df.groupby("game_pk").size()
        seasons[y] = {
            "pitches": len(df), "games": int(df["game_pk"].nunique()),
            "first": str(gd.min().date()), "last": str(gd.max().date()),
            "pitchers": int(df["pitcher"].nunique()),
            "pitches/game": float(per_game.mean()),
            "games <150 pitches": int((per_game < 150).sum()),
            "non-R rows": int((df["game_type"] != "R").sum()),
            "dup key (season file)": int(df.duplicated(KEY).sum()),
            "dup key (raw chunks)": raw_duplicates(data_dir, y),
            "null key rows": int(df[KEY].isna().any(axis=1).sum()),
            "LHP %": float((df["p_throws"] == "L").mean() * 100),
            "target null % (modeled pitch types)": float(typed["target"].isna().mean() * 100),
            "mean target x100": float(typed["target"].mean() * 100),
            "mean delta_run_exp x100": float(pd.to_numeric(typed["delta_run_exp"]).mean() * 100),
            "r(target, delta_run_exp)": float(typed["target"].corr(pd.to_numeric(typed["delta_run_exp"]).astype(float))),
            "excluded pitch types %": float(df["pitch_type"].isin(EXCLUDED).mean() * 100),
        }
        nulls[y] = df[[c for c in INPUTS if c in df.columns]].isna().mean() * 100
        nulls[y]["events (in play only)"] = float(
            df.loc[df["description"] == "hit_into_play", "events"].isna().mean() * 100)
        nulls[y]["xwOBA (in play only)"] = float(
            df.loc[df["description"] == "hit_into_play", "estimated_woba_using_speedangle"].isna().mean() * 100)
        mix[y] = df["pitch_type"].fillna("(null)").value_counts(normalize=True) * 100
        phys[y] = physics_check(df)
        print(f"{y}: {len(df):,} pitches")

    seasons_df = pd.DataFrame(seasons)
    nulls_df = pd.DataFrame(nulls)
    mix_df = pd.DataFrame(mix).fillna(0)
    order = [p for p in PITCH_TYPES if p in mix_df.index] + \
            sorted(p for p in mix_df.index if p not in PITCH_TYPES)
    mix_df = mix_df.loc[order]

    names = [k for k, v in phys[args.years[0]].items() if isinstance(v, dict) and "median" in v]
    phys_rows = {}
    for n in names:
        for stat in ("median", "p99"):
            phys_rows[f"{n} — {stat}"] = {y: phys[y][n][stat] for y in args.years}
    phys_df = pd.DataFrame(phys_rows).T
    impl_df = pd.DataFrame({y: {**phys[y]["implausible_rows"],
                                "non-finite grid angles": phys[y]["grid_nonfinite_rows"],
                                "rows checked": phys[y]["n"]} for y in args.years})
    sane_df = pd.DataFrame({y: {
        "vaa_ln 0.1% / 50% / 99.9%": " / ".join(f"{v:.1f}" for v in phys[y]["vaa_ln_range"]),
        "haa_ln 0.1% / 50% / 99.9%": " / ".join(f"{v:.1f}" for v in phys[y]["haa_ln_range"]),
        "FF mean vaa_ln": f"{phys[y]['ff_vaa_ln_mean']:.2f}",
        "FF mean haa_ln, LHP / RHP": " / ".join(f"{phys[y]['ff_haa_ln_mean_by_hand'][h]:.2f}" for h in "LR"),
    } for y in args.years})

    out = HERE / "reports"
    out.mkdir(exist_ok=True)
    md = ["# Statcast data-quality report", "",
          f"Regular seasons {args.years[0]}–{args.years[-1]}, from `fetch_data.py` "
          f"(pybaseball {env_info()['pybaseball']}). Generated by `data_quality.py`.", "",
          "## Rows per season", "", md_table(seasons_df, "{:.3f}"), "",
          "## Null rate (%) per input per season", "", md_table(nulls_df, "{:.2f}"), "",
          "## Pitch-type mix (%) by season", "", md_table(mix_df, "{:.2f}"), "",
          "## Physics check on real rows", "",
          "Absolute differences between `physics.py` and Statcast's own columns.", "",
          md_table(phys_df, "{:.4f}"), "", md_table(sane_df), "",
          "Rows failing plausibility bounds:", "", md_table(impl_df), ""]
    (out / "data_quality.md").write_text("\n".join(md))
    with open(out / "data_quality.json", "w") as f:
        json.dump({"env": env_info(), "seasons": seasons, "null_pct": nulls_df.to_dict(),
                   "pitch_mix_pct": mix_df.to_dict(), "physics": phys, "dtypes": dtypes},
                  f, indent=2, default=float)
    print(f"wrote {out/'data_quality.md'}")


if __name__ == "__main__":
    main()
