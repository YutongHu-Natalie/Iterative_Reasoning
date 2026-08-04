"""Build a fixed random annotation sample from the step-numbered DeepTheorem subset.

Selects 10,000 rows (out of 62,377) and assigns each a task_type:
  - positive    (4000)  unchanged proof, expected SUFFICIENT=1 / ERROR_TYPE=none
  - mistake     (2500)  annotator introduces a logical/computational mistake
  - gap         (2500)  annotator introduces a missing-justification gap
  - truncation  (1000)  annotator truncates the proof partway through

The assignment is written to Data/annotation_sample.json as {id: task_type},
keyed by the original DeepTheorem `id` (not row index), so it can later be
used to exclude these ids from other train/eval splits and prevent leakage.

Deterministic given SEED; re-running regenerates the same sample.
"""
import argparse
import json
import random

from datasets import load_from_disk

SEED = 42
BUCKET_SIZES = {
    "positive": 4000,
    "mistake": 2500,
    "gap": 2500,
    "truncation": 1000,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="./Data/Processed_DeepTheorem_StepNumbered")
    parser.add_argument("--out", default="./Data/annotation_sample.json")
    args = parser.parse_args()

    dt = load_from_disk(args.dataset)
    if hasattr(dt, "keys"):
        dt = dt["train"]

    ids = list(dt["id"])
    total_needed = sum(BUCKET_SIZES.values())
    assert len(ids) >= total_needed, f"only {len(ids)} rows available, need {total_needed}"

    rng = random.Random(SEED)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    sample_ids = shuffled[:total_needed]

    assignment = {}
    cursor = 0
    for task_type, size in BUCKET_SIZES.items():
        for i in sample_ids[cursor:cursor + size]:
            assignment[str(i)] = task_type
        cursor += size

    with open(args.out, "w") as f:
        json.dump(assignment, f, indent=1)

    print(f"Wrote {len(assignment)} assignments to {args.out}")
    for task_type, size in BUCKET_SIZES.items():
        print(f"  {task_type}: {size}")


if __name__ == "__main__":
    main()
