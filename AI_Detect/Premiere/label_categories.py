"""LLM-label the problems build_full_data.py could not categorize.

Reads Data/category_pending.json (problems with no source label, an
ambiguous one like HARP "precalculus", or several Putnam tags) and asks a
local LLM to pick ONE category from that problem's allowed options. Writes
Data/category_llm_labels.json; rerun build_full_data.py afterwards to merge.

Resumable: problems already labeled with a valid category are skipped.

Usage (on the cluster):
  python AI_Detect/Premiere/label_categories.py                    # Qwen3-8B, no thinking
  python AI_Detect/Premiere/label_categories.py --model qwen3_8b_think
  python AI_Detect/Premiere/build_full_data.py                     # merge labels
"""
import argparse
import json
import re
from pathlib import Path

from tqdm import tqdm

from run_detection import LLM_MODELS, atomic_write_json, extract_json, generate_gemma4, generate_qwen3, load_llm

HERE = Path(__file__).resolve().parent
PENDING_PATH = HERE / "Data" / "category_pending.json"
LABELS_PATH = HERE / "Data" / "category_llm_labels.json"

DEFINITIONS = {
    "Algebra": "polynomials, functional equations, inequalities, sequences and recurrences, "
               "algebraic identities, linear or abstract algebra",
    "Combinatorics": "counting, graphs, games and strategies, invariants, extremal and "
                     "combinatorial geometry (point sets, tilings, colorings), probability",
    "Geometry": "Euclidean geometry: triangles, circles, angles, lengths, concurrency and "
                "collinearity, including vector, complex or trigonometric proofs of such facts",
    "Number Theory": "divisibility, primes, congruences, Diophantine equations, digits, "
                     "integer-valued functions",
    "Analysis": "limits, continuity, derivatives, integrals, infinite series, real functions "
                "(mainly university contests such as Putnam, IMC)",
}

PROMPT = """Classify the following competition math problem into exactly ONE of these categories:

{options}

Choose the category of the main idea needed to solve the problem, not of surface words in the statement.

## Problem
{problem}

Respond with ONLY a JSON object, no other text:
{{"category": "<one of: {names}>"}}"""


def build_prompt(problem, allowed):
    options = "\n".join(f"- {c}: {DEFINITIONS[c]}" for c in allowed)
    return PROMPT.format(options=options, problem=problem, names=", ".join(allowed))


def parse_category(answer, allowed):
    parsed, _ = extract_json(answer)
    value = parsed.get("category") if parsed else None
    if not isinstance(value, str):
        # Fall back to a bare category name in the text.
        hits = [c for c in allowed if re.search(rf"\b{c}\b", answer or "", re.I)]
        return hits[0] if len(hits) == 1 else None
    value = value.strip().lower()
    return next((c for c in allowed if c.lower() == value), None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(LLM_MODELS), default="qwen3_8b_nothink")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="Defaults to 4096 with thinking, else 64.")
    parser.add_argument("--attempts", type=int, default=2, help="Retries when the answer is not a valid category.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=20)
    args = parser.parse_args()

    with open(PENDING_PATH, encoding="utf-8") as f:
        pending = json.load(f)
    labels = json.load(open(LABELS_PATH, encoding="utf-8")) if LABELS_PATH.exists() else {}
    todo = [(puid, p) for puid, p in pending.items()
            if labels.get(puid, {}).get("category") not in p["allowed"]]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(pending)} pending problems, {len(todo)} to label with {args.model}")
    if not todo:
        return

    cfg = LLM_MODELS[args.model]
    max_new_tokens = args.max_new_tokens or (4096 if cfg["thinking"] else 64)
    tok_or_proc, model = load_llm(cfg)
    generate = generate_qwen3 if cfg["family"] == "qwen3" else generate_gemma4

    for n, (puid, p) in enumerate(tqdm(todo, desc="labeling"), 1):
        prompt = build_prompt(p["problem"], p["allowed"])
        category, answers = None, []
        for _ in range(args.attempts):
            _, answer, _ = generate(tok_or_proc, model, prompt, cfg["thinking"], max_new_tokens)
            answers.append(answer)
            category = parse_category(answer, p["allowed"])
            if category:
                break
        labels[puid] = {"category": category, "allowed": p["allowed"], "category_raw": p["category_raw"],
                        "model": args.model, "raw_responses": answers}
        if n % args.save_every == 0:
            atomic_write_json(LABELS_PATH, labels)
    atomic_write_json(LABELS_PATH, labels)

    failed = [puid for puid, _ in todo if not labels[puid]["category"]]
    print(f"Labeled {len(todo) - len(failed)}/{len(todo)} -> {LABELS_PATH}"
          + (f"; no valid category for {len(failed)} (rerun to retry)" if failed else ""))
    print("Now rerun build_full_data.py to merge the labels.")


if __name__ == "__main__":
    main()
