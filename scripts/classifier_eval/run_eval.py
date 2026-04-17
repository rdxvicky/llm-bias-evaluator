"""
Run gemma3n:e2b on holdout.json and write predictions to classifier_preds.json.

Prerequisites:
  1. python scripts/make_splits.py          (creates holdout.json)
  2. ollama serve                           (Ollama must be running)
  3. ollama pull gemma3n:e2b               (model must be available)

Usage:
  python scripts/classifier_eval/run_eval.py
  python scripts/classifier_eval/run_eval.py --delay 0.2   # seconds between calls
"""
import argparse, json, sys, time
from pathlib import Path

# Allow running as a script from the repo root
sys.path.insert(0, str(Path(__file__).parent))
from classifier import classify, argmax_label

ROOT = Path(__file__).resolve().parent.parent.parent
HOLDOUT = ROOT / "holdout.json"
OUT = ROOT / "classifier_preds.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delay", type=float, default=0.1,
                        help="seconds to sleep between Ollama calls (default: 0.1)")
    args = parser.parse_args()

    if not HOLDOUT.exists():
        sys.exit("holdout.json not found — run  python scripts/make_splits.py  first.")

    holdout = json.loads(HOLDOUT.read_text())
    preds: list[dict] = []
    parse_errors = 0

    print(f"Evaluating {len(holdout)} hold-out samples with gemma3n:e2b …\n")
    for i, item in enumerate(holdout, 1):
        try:
            scores = classify(item["text"])
            pred = argmax_label(scores)
            preds.append({
                "id": item["id"],
                "text": item["text"],
                "true": item["true_category"],
                "pred": pred,
                "bias_scores": scores,
            })
        except Exception as exc:
            parse_errors += 1
            preds.append({
                "id": item["id"],
                "true": item["true_category"],
                "error": str(exc),
            })

        if i % 10 == 0 or i == len(holdout):
            ok = i - parse_errors
            print(f"  {i:>3}/{len(holdout)}  success={ok}  parse_errors={parse_errors}")

        if args.delay > 0:
            time.sleep(args.delay)

    OUT.write_text(json.dumps(preds, indent=2))
    success = len(preds) - parse_errors
    print(f"\nDone.  {success}/{len(holdout)} successful  |  {parse_errors} parse errors")
    if parse_errors:
        rate = parse_errors / len(holdout) * 100
        print(f"Parse-error rate: {rate:.1f}%  — review 'error' entries in {OUT.name}")
    print(f"Saved → {OUT}")


if __name__ == "__main__":
    main()
