"""
Stratified 70/30 train / hold-out split of dataset.json.
Run once:  python scripts/make_splits.py
Outputs:   train.json  holdout.json  (committed for reproducibility)
"""
import json, random
from collections import defaultdict
from pathlib import Path

random.seed(42)

root = Path(__file__).parent.parent
data = json.loads((root / "dataset.json").read_text())

by_cat: dict[str, list] = defaultdict(list)
for item in data:
    by_cat[item["true_category"]].append(item)

train, holdout = [], []
for cat in sorted(by_cat):
    items = by_cat[cat][:]
    random.shuffle(items)
    k = max(1, round(0.3 * len(items)))   # 30% hold-out
    holdout.extend(items[:k])
    train.extend(items[k:])

(root / "holdout.json").write_text(json.dumps(holdout, indent=2))
(root / "train.json").write_text(json.dumps(train, indent=2))

print(f"train={len(train)}  holdout={len(holdout)}")
print("\nHold-out breakdown (category : n):")
hcount: dict[str, int] = defaultdict(int)
for item in holdout:
    hcount[item["true_category"]] += 1
for cat in sorted(hcount):
    print(f"  {cat:<26}: {hcount[cat]}")
