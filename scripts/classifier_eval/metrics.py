"""
Compute per-class P/R/F1, confusion matrix, and bootstrap 95% CI for macro-F1.
Reads classifier_preds.json produced by run_eval.py.

Usage:
  python scripts/classifier_eval/metrics.py
  python scripts/classifier_eval/metrics.py --preds path/to/classifier_preds.json
"""
import argparse, json, sys
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
from sklearn.utils import resample

ROOT = Path(__file__).resolve().parent.parent.parent


def bootstrap_macro_f1(
    y_true: list[str], y_pred: list[str], n: int = 2000, seed: int = 42
) -> tuple[float, float]:
    """Return (lower, upper) 95% CI for macro-F1 via bootstrapping."""
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        yt = [y_true[i] for i in idx]
        yp = [y_pred[i] for i in idx]
        scores.append(f1_score(yt, yp, average="macro", zero_division=0))
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return float(lo), float(hi)


def print_confusion_matrix(y_true: list[str], y_pred: list[str]) -> None:
    cats = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=cats)
    col_w = max(len(c) for c in cats) + 2

    # Header
    row_label_w = 26
    print(" " * row_label_w + "  ".join(f"{c[:8]:>8}" for c in cats))
    for cat, row in zip(cats, cm):
        cells = "  ".join(f"{n:>8}" for n in row)
        print(f"  {cat:<{row_label_w - 2}}{cells}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds", default=str(ROOT / "classifier_preds.json"))
    parser.add_argument("--bootstrap-n", type=int, default=2000,
                        help="bootstrap iterations for CI (default: 2000)")
    args = parser.parse_args()

    preds_path = Path(args.preds)
    if not preds_path.exists():
        sys.exit(f"{preds_path} not found — run run_eval.py first.")

    all_preds = json.loads(preds_path.read_text())
    valid = [p for p in all_preds if "pred" in p]
    errors = [p for p in all_preds if "error" in p]

    # ── Parse-error report (Table E) ─────────────────────────────────────────
    print("=" * 65)
    print("TABLE E — PARSE-ERROR RATE")
    print("=" * 65)
    total = len(all_preds)
    print(f"  Total samples : {total}")
    print(f"  Valid          : {len(valid)}")
    print(f"  Parse errors   : {len(errors)}  ({len(errors)/total*100:.1f}%)")
    if errors:
        print("\n  Error details:")
        for e in errors[:10]:          # show at most 10
            print(f"    id={e['id']}  true={e['true']:<26}  {e['error'][:60]}")
        if len(errors) > 10:
            print(f"    … and {len(errors)-10} more")

    if not valid:
        sys.exit("No valid predictions to analyse.")

    y_true = [p["true"] for p in valid]
    y_pred = [p["pred"] for p in valid]

    # ── Table A — Per-class P/R/F1 ───────────────────────────────────────────
    print("\n" + "=" * 65)
    print("TABLE A — PER-CATEGORY PRECISION / RECALL / F1")
    print("=" * 65)
    print(classification_report(y_true, y_pred, digits=3))

    # ── Table B — Confusion matrix ───────────────────────────────────────────
    print("=" * 65)
    print("TABLE B — CONFUSION MATRIX  (rows=true, cols=predicted)")
    print("=" * 65)
    print_confusion_matrix(y_true, y_pred)

    # ── Headline metrics with bootstrap CI ──────────────────────────────────
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    lo, hi = bootstrap_macro_f1(y_true, y_pred, n=args.bootstrap_n)

    print(f"\n{'='*65}")
    print("HEADLINE METRICS")
    print(f"{'='*65}")
    print(f"  Accuracy      : {acc:.3f}")
    print(f"  Macro-F1      : {macro:.3f}  (95% CI [{lo:.3f}, {hi:.3f}], "
          f"n_boot={args.bootstrap_n})")
    print(f"  Weighted-F1   : {weighted:.3f}")
    n = len(valid)
    print(f"\n  n={n} valid predictions.  "
          f"CIs are wide at this sample size — report them honestly.")


if __name__ == "__main__":
    main()
