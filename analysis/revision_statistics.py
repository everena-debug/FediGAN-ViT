"""Reproduce the revision analyses (paper Tables 13 and 14) from the saved per-image predictions.

No GPU and no training are needed: the script reads the prediction files (*.npz) and run logs (*.json)
written by the notebook to fedigan_outputs/runs_paper_fast/.

Usage:
    python analysis/revision_statistics.py results/runs_paper_fast [output_dir]

Outputs (CSV):
    table13_per_seed_paired.csv   per-seed macro-F1, matched-seed differences, t-based and paired-bootstrap 95% CIs
    table14_per_class.csv         per-class recall (mean ± SD over seeds) and Glaucoma/Normal confusions
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import confusion_matrix, f1_score, recall_score

CLASSES = ["Cataract", "DR", "Glaucoma", "Normal"]
SEEDS = [42, 43, 44]
BOOT_RESAMPLES, BOOT_SEED = 2000, 0


def load_pred(runs, key, seed):
    z = np.load(os.path.join(runs, f"{key}__s{seed}.npz"))
    return z["y"], z["prob"]          # prob: (n, 4); for local-only runs (3 clients, n, 4)


def macro_f1(y, prob):
    if prob.ndim == 3:                # local-only: mean over the clients' own models
        return float(np.mean([f1_score(y, p.argmax(1), average="macro") for p in prob]))
    return float(f1_score(y, prob.argmax(1), average="macro"))


def logged_macro_f1(runs, key, seed):
    with open(os.path.join(runs, f"{key}__s{seed}.json")) as f:
        return json.load(f)["test"]["f1_macro"]


# The paper's intervals were drawn from two random streams (seed 0), each shared by two comparisons in this order.
BOOT_STREAMS = [["fedprox_none", "fedprox_oversample"], ["central_none", "local_gan"]]
_RNG = {}


def _stream_rng(key):
    for i, group in enumerate(BOOT_STREAMS):
        if key in group:
            if i not in _RNG:
                _RNG[i] = np.random.default_rng(BOOT_SEED)
            return _RNG[i]
    return np.random.default_rng(BOOT_SEED)


def paired_bootstrap(runs, other):
    """Mean over seeds of (proposed - other) macro-F1 on resampled test images; 95% percentile interval."""
    data = [(load_pred(runs, "proposed", s), load_pred(runs, other, s)) for s in SEEDS]
    n = len(data[0][0][0])
    rng = _stream_rng(other)
    diffs = []
    for _ in range(BOOT_RESAMPLES):
        i = rng.integers(0, n, n)
        diffs.append(np.mean([macro_f1(a[0][i], a[1][i] if a[1].ndim == 2 else a[1][:, i]) -
                              macro_f1(b[0][i], b[1][i] if b[1].ndim == 2 else b[1][:, i]) for a, b in data]))
    d = np.array(diffs) * 100
    return np.percentile(d, 2.5), np.percentile(d, 97.5)


def table13(runs):
    three_seed = [("No augmentation", "fedprox_none"), ("Random oversampling", "fedprox_oversample"),
                  ("Centralised ViT", "central_none"), ("Local-only ViT + cGAN", "local_gan")]
    one_seed = [("Partial participation", "partial"), ("Leave client 0 out", "loco_c0"),
                ("FedAvg + ViT + cGAN", "fedavg_gan"), ("FedProx + ResNet-50 + cGAN", "fedprox_gan_resnet50")]
    ours = {s: logged_macro_f1(runs, "proposed", s) * 100 for s in SEEDS}
    rows = []
    for name, key in three_seed:
        other = {s: logged_macro_f1(runs, key, s) * 100 for s in SEEDS}
        d = np.array([ours[s] - other[s] for s in SEEDS])
        h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
        lo, hi = paired_bootstrap(runs, key)
        rows.append({"Comparison": name, **{f"Seed {s} (ours / other)": f"{ours[s]:.2f} / {other[s]:.2f}" for s in SEEDS},
                     "Mean delta (points)": f"{d.mean():+.2f}", "95% CI over seeds": f"[{d.mean() - h:+.2f}, {d.mean() + h:+.2f}]",
                     "Paired bootstrap 95% CI": f"[{lo:+.2f}, {hi:+.2f}]"})
    for name, key in one_seed:
        o = logged_macro_f1(runs, key, 42) * 100
        rows.append({"Comparison": name, "Seed 42 (ours / other)": f"{ours[42]:.2f} / {o:.2f}", "Seed 43 (ours / other)": "-",
                     "Seed 44 (ours / other)": "-", "Mean delta (points)": f"{ours[42] - o:+.2f}",
                     "95% CI over seeds": "-", "Paired bootstrap 95% CI": "-"})
    return pd.DataFrame(rows)


def table14(runs):
    configs = [("FedProx + ViT, no augmentation", "fedprox_none"), ("FedProx + ViT + random oversampling", "fedprox_oversample"),
               ("FediGAN-ViT (cGAN)", "proposed"), ("Centralised ViT", "central_none")]
    rows = []
    for name, key in configs:
        rec, g2n, n2g, n_g, n_n = [], 0, 0, 0, 0
        for s in SEEDS:
            y, p = load_pred(runs, key, s)
            pred = p.argmax(1)
            rec.append(recall_score(y, pred, average=None, labels=range(4)) * 100)
            cm = confusion_matrix(y, pred, labels=range(4))
            g2n += cm[2, 3]; n2g += cm[3, 2]; n_g += cm[2].sum(); n_n += cm[3].sum()
        rec = np.array(rec)
        rows.append({"Configuration": name,
                     **{c: f"{rec[:, i].mean():.1f} ± {rec[:, i].std(ddof=1):.1f}" for i, c in enumerate(CLASSES)},
                     "Glaucoma -> Normal": f"{g2n} / {n_g}", "Normal -> Glaucoma": f"{n2g} / {n_n}"})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    runs = sys.argv[1] if len(sys.argv) > 1 else "results/runs_paper_fast"
    out = sys.argv[2] if len(sys.argv) > 2 else "results/tables"
    os.makedirs(out, exist_ok=True)
    t13, t14 = table13(runs), table14(runs)
    t13.to_csv(os.path.join(out, "table13_per_seed_paired.csv"), index=False)
    t14.to_csv(os.path.join(out, "table14_per_class.csv"), index=False)
    pd.set_option("display.width", 250)
    print("Table 13\n", t13.to_string(index=False), "\n\nTable 14\n", t14.to_string(index=False))
