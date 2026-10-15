"""
Run the local Stuff+ website:  python serve.py   ->  http://127.0.0.1:8000

On first run it builds the leaderboards (webapp/build.py). With data/ and the
shipped models in models/ (pipeline/final.py), it builds one bundle per frozen model
(Stuff+ (no remembering), Stuff+ (remembering), S6; whichever are present) and
the page's model menu switches between them. It opens on Stuff+ (no remembering):
its published leaderboards, every season and the grader out-of-fold. With data/
only, it trains a variant; with neither, it serves a synthetic demo.

    python -m webapp.folds --model models/stuffplus_v1.0.0.json
                                           # only for a v1.0.0 model; v1.1.0+ needs no extra step
    python serve.py --synthetic            # force the demo data
    python serve.py --rebuild              # rebuild after new data / code changes
    python serve.py --variant B_plus_center --train 2022 2023 2024 2025   # train instead
"""
import argparse
import webbrowser
from pathlib import Path

import uvicorn

from stuffplus.features import VARIANTS
from webapp.app import create_app
from webapp.build import build, bundle_path, site_models

ROOT = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=list(VARIANTS),
                    help="train this feature-set variant instead of using the frozen model")
    ap.add_argument("--train", nargs="*", type=int, help="training seasons (default: all)")
    ap.add_argument("--synthetic", action="store_true", help="use synthetic demo data")
    ap.add_argument("--rebuild", action="store_true", help="retrain even if a model bundle exists")
    ap.add_argument("--bundle", type=Path, nargs="+",
                    help="serve these models/*.joblib bundles (the first is the default)")
    ap.add_argument("--spread", default="none", choices=["keep", "center", "none"],
                    help="show Stuff+ with an SD of 10 per pitch type (keep type means / center at 100) "
                         "or exactly as the model outputs it (none); needs --rebuild to change")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()

    paths = a.bundle
    spread = None if a.spread == "none" else a.spread
    if paths is None:
        synthetic = a.synthetic or not any((ROOT / "data").glob("statcast_*.parquet"))
        if synthetic and not a.synthetic:
            print("No data/ found: serving SYNTHETIC demo data. Run `python -m pipeline.fetch_data` "
                  "then `python serve.py --rebuild` for real leaderboards.")
        variant = a.variant or "A_tj_v3"
        metas = [] if (synthetic or a.variant or a.train) else site_models()
        paths = []
        for frozen in metas or [None]:
            path = bundle_path(frozen.stem if frozen else variant, synthetic)
            if a.rebuild or a.train or not path.exists():
                path = build(variant, synthetic, a.train, frozen=frozen, spread=spread)
            paths.append(path)

    url = f"http://{a.host}:{a.port}"
    print(f"Serving {', '.join(p.name for p in paths)} at {url}")
    if not a.no_browser:
        webbrowser.open(url)
    uvicorn.run(create_app(paths), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
