"""Build a balanced (difficulty x category) sample from DeepTheorem for multi-model testing.

For each exact difficulty value, target = min(TARGET_PER_DIFFICULTY, rows available at that
difficulty). Within a difficulty we first try to take PER_CATEGORY rows from each of the 7
main categories (derived from the top-level prefix of domain[0]); any shortfall -- a category
with fewer than PER_CATEGORY rows, or leftover slots once every category is exhausted -- is
filled round-robin from whatever rows remain at that difficulty (any category, including the
long-tail "Other" bucket), so the target is always hit whenever the data allows it.

All original columns/metadata are preserved. Deterministic given SEED.
"""
import argparse
import json
import random
from collections import defaultdict

from datasets import load_from_disk

SEED = 42
TARGET_PER_DIFFICULTY = 14
PER_CATEGORY = 2

MAIN_CATEGORIES = [
    "Algebra",
    "Calculus",
    "Applied Mathematics",
    "Geometry",
    "Discrete Mathematics",
    "Linear Algebra",
    "Number Theory",
]


def top_level_category(domain):
    if not domain:
        return "Other"
    primary = domain[0].split(" -> ")[0].strip()
    return primary if primary in MAIN_CATEGORIES else "Other"


def select_for_difficulty(indices_by_category, target, per_category):
    selected = []
    used = set()

    # Pass 1: up to `per_category` from each main category.
    for cat in MAIN_CATEGORIES:
        pool = indices_by_category.get(cat, [])
        take = pool[:per_category]
        selected.extend(take)
        used.update(take)

    # Pass 2: fill remaining slots round-robin from whatever's left (any category).
    remaining_by_cat = {
        cat: [i for i in pool if i not in used]
        for cat, pool in indices_by_category.items()
    }
    active_cats = [c for c, pool in remaining_by_cat.items() if pool]

    while len(selected) < target and active_cats:
        for cat in list(active_cats):
            if len(selected) >= target:
                break
            pool = remaining_by_cat[cat]
            selected.append(pool.pop())
            if not pool:
                active_cats.remove(cat)

    return selected[:target]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="./Data/Processed_DeepTheorem")
    parser.add_argument("--out", default="./Data/Balanced_Model_Test_Set")
    parser.add_argument("--manifest", default="./Data/balanced_model_test_manifest.json")
    parser.add_argument("--target-per-difficulty", type=int, default=TARGET_PER_DIFFICULTY)
    parser.add_argument("--per-category", type=int, default=PER_CATEGORY)
    args = parser.parse_args()

    dt = load_from_disk(args.dataset)
    if hasattr(dt, "keys"):
        dt = dt["train"]

    rng = random.Random(SEED)

    by_difficulty = defaultdict(lambda: defaultdict(list))
    for idx, (diff, dom) in enumerate(zip(dt["difficulty"], dt["domain"])):
        by_difficulty[diff][top_level_category(dom)].append(idx)

    all_selected = []
    manifest = {}
    for diff in sorted(by_difficulty):
        indices_by_category = by_difficulty[diff]
        for pool in indices_by_category.values():
            rng.shuffle(pool)

        total_available = sum(len(p) for p in indices_by_category.values())
        target = min(args.target_per_difficulty, total_available)

        selected = select_for_difficulty(indices_by_category, target, args.per_category)
        all_selected.extend(selected)

        counts = defaultdict(int)
        for idx in selected:
            counts[top_level_category(dt[idx]["domain"])] += 1
        manifest[str(diff)] = {
            "target": target,
            "total_available": total_available,
            "selected": len(selected),
            "category_counts": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        }

    balanced = dt.select(sorted(all_selected))
    balanced.save_to_disk(args.out)
    with open(args.manifest, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Selected {len(all_selected)} rows total -> {args.out}")
    for diff, info in manifest.items():
        print(
            f"  difficulty {diff}: {info['selected']}/{info['target']} "
            f"(available {info['total_available']}) {info['category_counts']}"
        )


if __name__ == "__main__":
    main()
