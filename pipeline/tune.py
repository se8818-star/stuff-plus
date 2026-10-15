"""
Stage 2 regularization search (docs/PREREGISTRATION.md). Run only for A and the
stage 1 winner, with the same budget and search space:

    python -m pipeline.tune A
    python -m pipeline.tune JGH

- Optuna TPE (seed 0), 30 trials
- 3-fold GroupKFold by pitcher on the training years only, early stopping
  (patience 50, at most 3000 trees)
- regularizing parameters only: min_child_samples, num_leaves, reg_lambda,
  learning_rate (tree count set by early stopping)
- objective: out-of-fold loss on the model's own target. For the component
  model one parameter set is shared by the four models and the objective is the
  mean of each component's loss relative to its loss at TJ's settings.

Writes results/tuned_<name>.json. Whether the tuned settings are adopted is
decided afterwards by the selection score (pipeline/develop.py --params ...).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

from stuffplus import run_ablation as ra
from pipeline.develop import TRAIN_YEARS, load_dev
from stuffplus.models import CANDIDATES, TJ_PARAMS, component_tasks

ROOT = Path(__file__).resolve().parent.parent
N_TRIALS, N_FOLDS, PATIENCE, MAX_TREES = 30, 3, 50, 3000


def tasks_for(spec, tr):
    """(name, rows, label, is_classifier) for every model the candidate fits."""
    feats = spec["features"]
    if spec["kind"] == "component":
        return component_tasks(tr.dropna(subset=feats + ["target", "is_swing"]))
    return [("direct", tr.dropna(subset=feats + [spec["target"]]), spec["target"], False)]


def cv_loss(rows, feats, label, clf, params, early_stopping):
    """Mean held-out loss (MSE or log loss) over GroupKFold by pitcher, and the
    mean number of trees used."""
    import lightgbm as lgb
    losses, trees = [], []
    for i_fit, i_val in GroupKFold(n_splits=N_FOLDS).split(rows, groups=rows["pitcher"]):
        fit, val = rows.iloc[i_fit], rows.iloc[i_val]
        m = (lgb.LGBMClassifier if clf else lgb.LGBMRegressor)(**{**TJ_PARAMS, **params, "random_state": 0})
        kw = dict(eval_set=[(val[feats], val[label].to_numpy())],
                  callbacks=[lgb.early_stopping(PATIENCE, verbose=False)]) if early_stopping else {}
        m.fit(fit[feats], fit[label].to_numpy(), **kw)
        y = val[label].to_numpy()
        if clf:
            p = np.clip(m.predict_proba(val[feats])[:, 1], 1e-7, 1 - 1e-7)
            losses.append(float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()))
        else:
            losses.append(float(((y - m.predict(val[feats])) ** 2).mean()))
        trees.append(m.best_iteration_ if early_stopping and m.best_iteration_ else m.n_estimators)
    return float(np.mean(losses)), int(round(np.mean(trees)))


def suggest(trial):
    return {"min_child_samples": trial.suggest_int("min_child_samples", 20, 2000, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 7, 127, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 100, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True)}


def main():
    import optuna
    ap = argparse.ArgumentParser()
    ap.add_argument("name", choices=list(CANDIDATES))
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    args = ap.parse_args()
    assert args.trials <= N_TRIALS, "the pre-registered budget is 30 trials"

    t0 = time.time()
    spec = CANDIDATES[args.name]
    df = load_dev()
    tr = df[df["year"].isin(TRAIN_YEARS)]             # training years only
    tasks = tasks_for(spec, tr)
    feats = spec["features"]
    base = {n: cv_loss(rows, feats, label, clf, {}, early_stopping=False)[0]
            for n, rows, label, clf in tasks}
    print(f"{args.name}: loss at TJ's settings {base} ({time.time() - t0:.0f}s)", flush=True)

    def objective(trial):
        params = {**suggest(trial), "n_estimators": MAX_TREES}
        rel, trees = [], {}
        for n, rows, label, clf in tasks:
            loss, trees[n] = cv_loss(rows, feats, label, clf, params, early_stopping=True)
            rel.append(loss / base[n])
        trial.set_user_attr("trees", trees)
        print(f"  trial {trial.number}: {np.mean(rel):.6f} {trial.params} trees {trees} "
              f"({time.time() - t0:.0f}s)", flush=True)
        return float(np.mean(rel))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=args.trials)
    best, trees = study.best_trial, study.best_trial.user_attrs["trees"]
    params = dict(best.params)
    if spec["kind"] == "component":
        params["n_estimators_by_component"] = trees
    else:
        params["n_estimators"] = trees["direct"]
    out = ROOT / "results" / f"tuned_{args.name}.json"
    out.write_text(json.dumps({
        "name": args.name, "params": params, "relative_loss": best.value, "tj_loss": base,
        "trials": [{"number": t.number, "value": t.value, "params": t.params,
                    "trees": t.user_attrs.get("trees")} for t in study.trials],
        "budget": {"trials": args.trials, "folds": N_FOLDS, "patience": PATIENCE, "max_trees": MAX_TREES},
        "train_years": TRAIN_YEARS, "env": ra.env_info(), "seconds": time.time() - t0}, indent=1))
    print(f"best relative loss {best.value:.6f} with {params}; wrote {out.name} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
