"""Build the training corpus from the hand-curated seed taxonomy. stdlib only.

Merges src/seeds.jsonl with optional external corpora (e.g. UCI SMS spam), then
produces reproducible train/val/test splits. The test split must contain at
least a few examples of EVERY scam family, so category-level metrics are
meaningful.

The seeds are hand-curated and small by design — the notebook multiplies them
via paraphrase augmentation (documented there), so test rows stay unseen.

Usage:
    python -m src.build_dataset --seeds src/seeds.jsonl --out data/ [--seed 7]
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

REQUIRED_FIELDS = {"text", "label", "scam_type", "red_flags", "lang"}


def load_seeds(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            missing = REQUIRED_FIELDS - set(row)
            if missing:
                raise ValueError(f"seed line {i} missing fields: {missing}")
            if row["label"] not in {"scam", "benign"}:
                raise ValueError(f"seed line {i}: bad label {row['label']!r}")
            if row["label"] == "scam":
                # red_flags must be substrings of the text (used for highlighting)
                for flag in row["red_flags"]:
                    if flag not in row["text"]:
                        raise ValueError(f"seed line {i}: red_flag not in text: {flag!r}")
                if not row["scam_type"]:
                    raise ValueError(f"seed line {i}: scam rows need a scam_type")
            rows.append(row)
    return rows


def split(rows: list[dict], test_frac: float = 0.15, seed: int = 7) -> tuple[list, list, list]:
    """Stratified split: every scam family keeps >=2 test examples when possible."""
    rng = random.Random(seed)
    by_type: dict[str, list] = defaultdict(list)
    for row in rows:
        by_type[row["scam_type"] or "benign"].append(row)

    train, val, test = [], [], []
    for key, group in sorted(by_type.items()):
        rng.shuffle(group)
        n = len(group)
        n_test = max(2, round(n * test_frac)) if n >= 4 else (1 if n >= 2 else 0)
        n_val = max(1, round(n * 0.1)) if n >= 3 else 0
        test += group[:n_test]
        val += group[n_test : n_test + n_val]
        train += group[n_test + n_val :]
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="src/seeds.jsonl")
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rows = load_seeds(Path(args.seeds))
    train, val, test = split(rows, seed=args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as fh:
            for row in part:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    types = Counter(r["scam_type"] or r["label"] for r in rows)
    print(f"total={len(rows)}  train={len(train)}  val={len(val)}  test={len(test)}")
    print("taxonomy:", dict(sorted(types.items())))
    print(f"→ {out}/")


if __name__ == "__main__":
    main()
