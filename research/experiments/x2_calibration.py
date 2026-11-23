"""X2 - Does calibrating the Random Forest score fix the threshold problem?

Research-gap experiment (not part of the original paper). The detector's score is
a forest vote fraction, which the paper says is not a calibrated probability, and
its threshold cannot be chosen reliably on the validation block (60 benign flows).
This script tests a lightweight version of the proposed remedy: Platt scaling and
isotonic regression fitted on the chronological validation block, evaluated on the
chronological test partition.

Protocol (the same blocks as E12): the first 2,154,476 flows fit the forest, the
next 359,080 are the validation block used to fit the calibrators, and the last
30% of the corpus is the test partition. Only the cache written by E0/E16 is read.

Usage (from the repository root, XAISDN_DATA set):
    python research/experiments/x2_calibration.py
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "common"))

from paths import DATA_ROOT, RESULTS, SEED, log_event, save_result  # noqa: E402

CACHE = DATA_ROOT / "cache"
EPS = 1e-6


def reliability(y: np.ndarray, p: np.ndarray, bins: int = 10):
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece, table = 0.0, []
    for b in range(bins):
        m = idx == b
        c = int(m.sum())
        if c:
            ece += c / len(y) * abs(float(y[m].mean()) - float(p[m].mean()))
            table.append({"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": c,
                          "mean_score": float(p[m].mean()),
                          "attack_rate": float(y[m].mean())})
    return float(ece), table


def metrics(y: np.ndarray, p: np.ndarray, tau: float) -> dict:
    pred = (p >= tau).astype(np.int8)
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    tp = int(((pred == 1) & (y == 1)).sum())
    pc = np.clip(p, EPS, 1 - EPS)
    ece, table = reliability(y, p)
    return {
        "tau": tau,
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "fpr": fp / max(fp + tn, 1), "fnr": fn / max(fn + tp, 1),
        "fp": fp, "fn": fn,
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))),
        "ece_10bins": ece,
        "roc_auc": float(roc_auc_score(y, p)),
    }, table


def main() -> int:
    t_all = time.time()
    log_event("X2", "start")
    X = np.load(CACHE / "syn0311_X_repaired.npy")
    y = np.load(CACHE / "syn0311_y.npy").astype(np.int8)
    n = len(y)
    split = int(round(0.7 * n))
    val_n = int(math.ceil(0.10 * n))
    fit_end = split - val_n
    yfit, yval, yte = y[:fit_end], y[fit_end:split], y[split:]
    print(f"fit={fit_end:,} (benign {int((yfit == 0).sum()):,})  "
          f"validation={val_n:,} (benign {int((yval == 0).sum()):,})  "
          f"test={n - split:,} (benign {int((yte == 0).sum()):,})", flush=True)

    if len(np.unique(yfit)) < 2 or len(np.unique(yte)) < 2:
        print("ERROR: fit or test blocks contain a single class; the cache looks wrong.",
              flush=True)
        return 1
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=200, max_features="sqrt", bootstrap=True,
                                class_weight="balanced", n_jobs=-1, random_state=SEED)
    rf.fit(X[:fit_end], yfit)
    print(f"forest fitted in {time.time() - t0:.0f}s", flush=True)
    s_val = rf.predict_proba(X[fit_end:split])[:, 1]
    s_te = rf.predict_proba(X[split:])[:, 1]
    del X

    platt = LogisticRegression(C=1.0, max_iter=1000).fit(s_val.reshape(-1, 1), yval)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(s_val, yval)
    p_platt = platt.predict_proba(s_te.reshape(-1, 1))[:, 1]
    p_iso = iso.predict(s_te)

    cases = [("raw vote fraction", s_te, 0.70, "raw@0.70 (paper)"),
             ("raw vote fraction", s_te, 0.50, "raw@0.50"),
             ("Platt scaling", p_platt, 0.50, "platt@0.50"),
             ("isotonic regression", p_iso, 0.50, "isotonic@0.50")]
    results, rel_rows = [], []
    for label, p, tau, tag in cases:
        m, table = metrics(yte, p, tau)
        m.update(method=label, tag=tag)
        results.append(m)
        for r in table:
            rel_rows.append({"method": label, "tau": tau, **r})
    df = pd.DataFrame(results)[["tag", "tau", "macro_f1", "fpr", "fnr", "fp", "fn",
                                "brier", "log_loss", "ece_10bins", "roc_auc"]]
    print("\nTEST PARTITION (chronological last 30%)")
    show = df.copy()
    for c in ("macro_f1", "fpr", "fnr"):
        show[c] = (show[c] * 100).round(4)
    for c in ("brier", "log_loss", "ece_10bins", "roc_auc"):
        show[c] = show[c].round(5)
    print(show.to_string(index=False))

    sanity = {}
    e12 = RESULTS / "E12_pr_threshold.json"
    if e12.exists():
        ref = json.loads(e12.read_text(encoding="utf-8")).get(
            "test_operating_point_at_submitted_tau_0.70", {})
        if "macro_f1" in ref:
            d = results[0]["macro_f1"] - ref["macro_f1"]
            sanity = {"raw_0.70_macro_f1": results[0]["macro_f1"],
                      "e12_macro_f1": ref["macro_f1"], "delta": d,
                      "within_0.002": abs(d) <= 0.002}
            print(f"\nSanity check: raw@0.70 macro F1 differs from E12 by {d:+.5f} "
                  f"({'OK' if abs(d) <= 0.002 else 'CHECK'})")

    pd.DataFrame(rel_rows).to_csv(RESULTS / "X2_reliability.csv", index=False)
    out = save_result("X2_calibration", {
        "experiment": "X2",
        "objective": "test whether Platt / isotonic calibration of the forest score "
                     "helps under the chronological protocol",
        "n_fit": fit_end, "n_validation": val_n, "n_test": n - split,
        "benign_validation": int((yval == 0).sum()), "benign_test": int((yte == 0).sum()),
        "attack_rate_validation": float(yval.mean()), "attack_rate_test": float(yte.mean()),
        "results": results, "sanity": sanity,
        "total_runtime_s": round(time.time() - t_all, 1)})
    log_event("X2", "done", result=str(out))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
