"""
Threshold sweep, ROC/PR curves, and calibration analysis for gemma3n:e2b.

Two modes:
  1. Bias (one-vs-rest per category) — runs on classifier_preds.json
  2. Mental-health binary — runs on mh_preds.json  (requires MH hold-out dataset)

Usage:
  # Bias threshold sweep (runs now)
  python scripts/classifier_eval/threshold_sweep.py --mode bias

  # MH sweep (requires mh_preds.json from a labelled MH dataset)
  python scripts/classifier_eval/threshold_sweep.py --mode mh

Outputs PNG files into results/plots/.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for CI
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, precision_recall_curve,
    auc, average_precision_score, roc_auc_score,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parent.parent.parent
PLOTS_DIR = ROOT / "results" / "plots"


# ── helpers ──────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, name: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path.relative_to(ROOT)}")


def plot_roc_pr(
    y_true: np.ndarray,
    y_score: np.ndarray,
    label: str,
    tau: float,
    filename_prefix: str,
) -> dict:
    """Plot ROC and PR curves, mark the operating point at threshold tau."""
    roc_auc = roc_auc_score(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)

    fpr, tpr, thr_roc = roc_curve(y_true, y_score)
    prec, rec, thr_pr = precision_recall_curve(y_true, y_score)

    # Operating point at tau
    op_idx = np.argmin(np.abs(thr_roc - tau))
    op_fpr, op_tpr = fpr[op_idx], tpr[op_idx]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(f"gemma3n:e2b  —  {label}", fontsize=12)

    # ROC
    ax = axes[0]
    ax.plot(fpr, tpr, lw=2, label=f"ROC-AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax.scatter([op_fpr], [op_tpr], s=80, zorder=5,
               color="crimson", label=f"τ={tau}  (FPR={op_fpr:.2f}, TPR={op_tpr:.2f})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("ROC Curve")
    ax.legend(fontsize=8)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])

    # PR
    ax = axes[1]
    ax.plot(rec, prec, lw=2, label=f"PR-AUC = {pr_auc:.3f}")
    baseline = y_true.mean()
    ax.axhline(baseline, color="k", ls="--", lw=0.8,
               alpha=0.5, label=f"Baseline (prev={baseline:.2f})")
    # Mark tau on PR
    pr_idx = np.argmin(np.abs(thr_pr - tau))
    ax.scatter([rec[pr_idx]], [prec[pr_idx]], s=80, zorder=5, color="crimson",
               label=f"τ={tau}  (P={prec[pr_idx]:.2f}, R={rec[pr_idx]:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(fontsize=8)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])

    _save(fig, f"{filename_prefix}_roc_pr.png")

    return {
        "label": label, "tau": tau,
        "roc_auc": round(roc_auc, 3), "pr_auc": round(pr_auc, 3),
        "operating_point": {
            "fpr": round(float(op_fpr), 3),
            "tpr_recall": round(float(op_tpr), 3),
            "precision": round(float(prec[pr_idx]), 3),
        },
    }


def plot_calibration(
    y_true: np.ndarray,
    y_score: np.ndarray,
    label: str,
    filename_prefix: str,
    n_bins: int = 10,
) -> dict:
    """Reliability diagram + ECE."""
    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=n_bins)
    ece = float(np.mean(np.abs(prob_true - prob_pred)))
    brier = brier_score_loss(y_true, y_score)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(prob_pred, prob_true, "o-", lw=2, label="gemma3n:e2b")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Calibration — {label}\nECE={ece:.3f}  Brier={brier:.3f}")
    ax.legend(fontsize=8)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    _save(fig, f"{filename_prefix}_calibration.png")

    return {"label": label, "ece": round(ece, 4), "brier": round(brier, 4)}


def tau_sweep_table(
    y_true: np.ndarray,
    y_score: np.ndarray,
    label: str,
    taus: np.ndarray | None = None,
) -> None:
    """Print a table of TP/FP/FN/TN at each threshold step."""
    if taus is None:
        taus = np.arange(0.05, 1.0, 0.05)
    print(f"\n  Threshold sweep — {label}")
    print(f"  {'τ':>5}  {'TP':>5}  {'FP':>5}  {'FN':>5}  {'TN':>5}  "
          f"{'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'FPR':>6}")
    for tau in taus:
        yp = (y_score >= tau).astype(int)
        tp = int(((yp == 1) & (y_true == 1)).sum())
        fp = int(((yp == 1) & (y_true == 0)).sum())
        fn = int(((yp == 0) & (y_true == 1)).sum())
        tn = int(((yp == 0) & (y_true == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        print(f"  {tau:>5.2f}  {tp:>5}  {fp:>5}  {fn:>5}  {tn:>5}  "
              f"{prec:>6.3f}  {rec:>6.3f}  {f1:>6.3f}  {fpr:>6.3f}")


# ── bias mode ────────────────────────────────────────────────────────────────

def run_bias(tau: float) -> None:
    preds_path = ROOT / "classifier_preds.json"
    if not preds_path.exists():
        sys.exit("classifier_preds.json not found — run run_eval.py first.")

    all_preds = json.loads(preds_path.read_text())
    valid = [p for p in all_preds if "bias_scores" in p]
    if not valid:
        sys.exit("No bias_scores found in classifier_preds.json.")

    cats = list(valid[0]["bias_scores"].keys())
    results = []

    print(f"\nBias threshold sweep  (τ={tau})  —  one-vs-rest per category\n")
    print(f"  {'Category':<26}  {'ROC-AUC':>8}  {'PR-AUC':>7}  {'ECE':>6}")
    print("  " + "-" * 54)

    for cat in cats:
        y_true = np.array([1 if p["true"] == cat else 0 for p in valid])
        y_score = np.array([p["bias_scores"][cat] for p in valid])

        if y_true.sum() == 0:
            print(f"  {cat:<26}  (no positives in holdout — skip)")
            continue

        r = plot_roc_pr(y_true, y_score, cat, tau,
                        filename_prefix=f"bias_{cat}")
        c = plot_calibration(y_true, y_score, cat,
                             filename_prefix=f"bias_{cat}")
        print(f"  {cat:<26}  {r['roc_auc']:>8.3f}  {r['pr_auc']:>7.3f}  {c['ece']:>6.4f}")
        tau_sweep_table(y_true, y_score, cat,
                        taus=np.array([0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]))
        results.append({**r, **c})

    out = ROOT / "results" / "bias_threshold_sweep.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nFull results → {out.relative_to(ROOT)}")
    print(f"Plots        → results/plots/")


# ── mental-health mode ───────────────────────────────────────────────────────

def run_mh(tau_sh: float, tau_distress: float) -> None:
    """
    Mental-health threshold sweep.

    Requires mh_preds.json with schema:
      [{"id": ..., "text": ..., "true_mh": "self_harm"|"severe_distress"|
        "existential_crisis"|"emotional_dependency"|"none",
        "mh_scores": {"self_harm": 0.0, "severe_distress": 0.0, ...}}, ...]

    Collect this dataset before running (see Step 5 in the classifier eval plan).
    Target: ~200 items (50 per signal + 50 benign), two raters, Cohen's κ ≥ 0.6.
    """
    mh_path = ROOT / "mh_preds.json"
    if not mh_path.exists():
        print(
            "\n[MH MODE] mh_preds.json not found.\n"
            "This sweep requires a labelled mental-health hold-out dataset.\n"
            "Steps to create it:\n"
            "  1. Collect ~200 samples (r/SuicideWatch, CLPsych, DAIC-WOZ excerpts + benign)\n"
            "  2. Two raters label each item; compute Cohen's κ — target ≥ 0.6\n"
            "  3. Get ethics sign-off before storing MH-flagged text on disk\n"
            "  4. Run gemma3n:e2b with the MH scoring prompt and save mh_preds.json\n"
            "  5. Re-run:  python scripts/classifier_eval/threshold_sweep.py --mode mh\n"
        )
        return

    all_preds = json.loads(mh_path.read_text())
    valid = [p for p in all_preds if "mh_scores" in p]
    mh_cats = ["self_harm", "severe_distress", "existential_crisis", "emotional_dependency"]
    tau_map = {"self_harm": tau_sh}   # use tau_distress for the rest

    print(f"\nMH threshold sweep  (self_harm τ={tau_sh}, others τ={tau_distress})\n")
    for cat in mh_cats:
        y_true = np.array([1 if p["true_mh"] == cat else 0 for p in valid])
        y_score = np.array([p["mh_scores"][cat] for p in valid])
        if y_true.sum() == 0:
            print(f"  {cat}: no positives — skip")
            continue
        tau = tau_map.get(cat, tau_distress)
        r = plot_roc_pr(y_true, y_score, cat, tau, filename_prefix=f"mh_{cat}")
        c = plot_calibration(y_true, y_score, cat, filename_prefix=f"mh_{cat}")
        tau_sweep_table(y_true, y_score, cat,
                        taus=np.array([0.40, 0.50, 0.60, 0.70, 0.80]))
        print(f"  {cat:<28}  ROC-AUC={r['roc_auc']:.3f}  ECE={c['ece']:.4f}")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["bias", "mh"], default="bias")
    parser.add_argument("--tau", type=float, default=0.70,
                        help="operating-point threshold to annotate on plots (default: 0.70)")
    parser.add_argument("--tau-distress", type=float, default=0.60,
                        help="threshold for non-self-harm MH signals (default: 0.60)")
    args = parser.parse_args()

    if args.mode == "bias":
        run_bias(tau=args.tau)
    else:
        run_mh(tau_sh=args.tau, tau_distress=args.tau_distress)


if __name__ == "__main__":
    main()
