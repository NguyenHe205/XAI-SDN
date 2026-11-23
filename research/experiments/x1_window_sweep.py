"""X1 - Sensitivity of the detector to the entropy window length N.

Research-gap experiment (not part of the original paper). The paper fixes
N = 1000 flows before any evaluation and states that no sweep over N was
performed (Section VII-O). This script runs that sweep under the paper's own
chronological 70/30 protocol and the repaired 88-dimensional feature set, plus a
no-entropy reference (the 80 CICFlowMeter features only).

Everything except the window length is held fixed: the same cleaning, the same
temporal split, the same Random Forest configuration (200 trees, balanced class
weights) and the same decision threshold tau = 0.70.

Usage (from the repository root, XAISDN_DATA set):
    python research/experiments/x1_window_sweep.py
    python research/experiments/x1_window_sweep.py --ns 250,500,1000,2000,4000 --seeds 42
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "common"))

import data as D  # noqa: E402
from paths import DATA_ROOT, RESULTS, SEED, SYN_0311, log_event, save_result  # noqa: E402

TAU = 0.70
CACHE = DATA_ROOT / "cache"
# macro F1 that E16 printed for the repaired feature set at N = 1000, seed 42
E16_REPAIRED_MACRO_F1 = 0.978180


def evaluate(y_true: np.ndarray, score: np.ndarray, tau: float = TAU) -> dict:
    pred = (score >= tau).astype(np.int8)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    return {
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
        "accuracy": (tp + tn) / len(y_true),
        "fpr": fp / max(fp + tn, 1),
        "fnr": fn / max(fn + tp, 1),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "roc_auc": float(roc_auc_score(y_true, score)),
        "avg_precision": float(average_precision_score(y_true, score)),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(SYN_0311))
    ap.add_argument("--ns", default="250,500,1000,2000,4000",
                    help="comma-separated window lengths")
    ap.add_argument("--seeds", default=str(SEED), help="comma-separated seeds")
    ap.add_argument("--trees", type=int, default=200)
    ap.add_argument("--max-rows", type=int, default=None, help="testing only")
    ap.add_argument("--name", default="X1_window_sweep")
    a = ap.parse_args(argv)
    ns = [int(x) for x in a.ns.split(",") if x.strip()]
    seeds = [int(x) for x in a.seeds.split(",") if x.strip()]
    log_event("X1", "start", ns=ns, seeds=seeds)
    t_all = time.time()

    print(f"Loading {a.data}", flush=True)
    df = D.load_partition(Path(a.data), nrows=a.max_rows)
    df, _ = D.clean_partition(df)
    print("Sorting temporally...", flush=True)
    df = D.sort_temporal(df)
    y = D.to_binary_label(df["Label"])
    n = len(y)
    print(f"  total={n:,} benign={int((y == 0).sum()):,} attack={int((y == 1).sum()):,}",
          flush=True)

    y_agree = None
    cy = CACHE / "syn0311_y.npy"
    if a.max_rows is None and cy.exists():
        cached = np.load(cy)
        y_agree = bool(len(cached) == n and np.array_equal(cached, y))
        print(f"  labels identical to the E0 cache: {y_agree}", flush=True)
        if not y_agree:
            print("  WARNING: row order differs from E0; results are not comparable "
                  "with E16/E17.", flush=True)

    streams, _ = D.entropy_streams(df, repaired=True)
    feat = [c for c in D.CIC_FEATURE_NAMES if c in df.columns]
    Xc = df[feat].to_numpy(dtype=np.float32)
    del df
    gc.collect()

    split = int(round(0.7 * n))
    tr, te = slice(0, split), slice(split, n)
    yte = y[te]
    if len(np.unique(y[tr])) < 2 or len(np.unique(yte)) < 2:
        print("ERROR: the training or test partition contains a single class; "
              "check that the full Syn.csv was loaded.", flush=True)
        return 1
    print(f"  train={split:,}  test={n - split:,}  test_benign={int((yte == 0).sum()):,}",
          flush=True)

    variants = [("cic_only", 0)] + [(f"N={N}", N) for N in ns]
    rows: list[dict] = []
    for name, N in variants:
        t0 = time.time()
        if N == 0:
            X, ent_s = Xc, 0.0
        else:
            H = D.rolling_entropy(streams, window=N).astype(np.float32)
            X = np.hstack([Xc, H])
            del H
            ent_s = time.time() - t0
        for seed in seeds:
            t1 = time.time()
            rf = RandomForestClassifier(
                n_estimators=a.trees, max_features="sqrt", bootstrap=True,
                class_weight="balanced", n_jobs=-1, random_state=seed)
            rf.fit(X[tr], y[tr])
            fit_s = time.time() - t1
            score = rf.predict_proba(X[te])[:, 1]
            m = evaluate(yte, score)
            m.update(variant=name, window=N, seed=seed, n_features=int(X.shape[1]),
                     entropy_s=round(ent_s, 1), fit_s=round(fit_s, 1))
            rows.append(m)
            print(f"  [{name:9s} seed={seed}] macroF1={m['macro_f1']:.5f} "
                  f"FPR={m['fpr']*100:.4f}% FNR={m['fnr']*100:.4f}% "
                  f"(entropy {ent_s:.0f}s, fit {fit_s:.0f}s)", flush=True)
            del rf, score
            gc.collect()
        if N != 0:
            del X
        gc.collect()

    res = pd.DataFrame(rows)
    summary = (res.groupby("variant", sort=False)
                  .agg(window=("window", "first"), n_features=("n_features", "first"),
                       macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
                       fpr_mean=("fpr", "mean"), fnr_mean=("fnr", "mean"),
                       fp_mean=("fp", "mean"), fn_mean=("fn", "mean"),
                       auc_mean=("roc_auc", "mean"))
                  .reset_index())
    summary["macro_f1_std"] = summary["macro_f1_std"].fillna(0.0)
    print("\nSUMMARY (chronological 70/30, tau = 0.70)")
    show = summary.copy()
    show["macro_f1_mean"] = (show["macro_f1_mean"] * 100).round(3)
    show["macro_f1_std"] = (show["macro_f1_std"] * 100).round(3)
    show["fpr_mean"] = (show["fpr_mean"] * 100).round(4)
    show["fnr_mean"] = (show["fnr_mean"] * 100).round(4)
    print(show[["variant", "n_features", "macro_f1_mean", "macro_f1_std",
                "fpr_mean", "fnr_mean", "fp_mean", "fn_mean"]].to_string(index=False))

    sanity = {}
    ref = res[(res["window"] == 1000) & (res["seed"] == SEED)]
    if len(ref):
        d = float(ref["macro_f1"].iloc[0]) - E16_REPAIRED_MACRO_F1
        sanity = {"n1000_seed42_macro_f1": float(ref["macro_f1"].iloc[0]),
                  "e16_repaired_macro_f1": E16_REPAIRED_MACRO_F1,
                  "delta": d, "within_0.002": abs(d) <= 0.002}
        print(f"\nSanity check: N=1000 macro F1 differs from E16 by {d:+.5f} "
              f"({'OK' if abs(d) <= 0.002 else 'CHECK: pipeline differs from E16'})")

    res.to_csv(RESULTS / f"{a.name}.csv", index=False)
    out = save_result(a.name, {
        "experiment": "X1",
        "objective": "entropy window length sensitivity (research-gap experiment)",
        "protocol": "chronological 70/30, repaired 88-dim feature set, tau = 0.70",
        "n_train": split, "n_test": n - split, "benign_in_test": int((yte == 0).sum()),
        "labels_match_E0_cache": y_agree, "seeds": seeds, "windows": ns,
        "n_trees": a.trees, "rows": rows,
        "summary": summary.to_dict(orient="records"), "sanity": sanity,
        "total_runtime_s": round(time.time() - t_all, 1)})
    log_event("X1", "done", result=str(out))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
