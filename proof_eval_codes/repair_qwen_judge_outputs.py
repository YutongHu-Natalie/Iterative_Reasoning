"""One-off repair pass for Results/eval_scores/qwen2.5_math_7b.json and
Results/eval_pass/qwen2.5_math_7b.json.

Qwen2.5-Math-7B answered most judge prompts in its trained \\boxed{}/\\text{}
LaTeX style instead of plain JSON, so nearly every row was previously stored
with parse_error=True and null fields even though the underlying judgment is
present in raw_response. eval_utils.extract_json() now has a repair pass for
this pattern -- this script re-normalizes every already-collected raw_response
against it:

  - rows that now parse are rewritten with their recovered fields (matching
    normalize_score_response/normalize_pass_response's own schema exactly, so
    this is not a new code path, just applying the current one retroactively)
  - rows that still fail to parse are dropped, not left in place, so that
    rerunning run_qwen2.5math_judge.py (resumable by (judged_run, id)) will
    regenerate exactly those rows -- now against the strengthened prompts in
    eval_prompts.py that explicitly forbid \\boxed{}/LaTeX -- instead of
    silently skipping them as already "done".

Safe to re-run; it only touches parse_error=True rows.
"""
from eval_utils import (
    EVAL_PASS_DIR,
    EVAL_SCORES_DIR,
    atomic_write_json,
    load_existing,
    normalize_pass_response,
    normalize_score_response,
)

JUDGE_SLUG = "qwen2.5_math_7b"

BASE_FIELDS = (
    "id", "domain", "difficulty", "truth_value", "source",
    "judged_run", "judged_model", "judged_thinking_mode", "judge_model",
)


def repair(path, normalize_fn):
    records = load_existing(path)
    kept = []
    recovered = 0
    dropped = 0
    for r in records:
        if not r.get("parse_error"):
            kept.append(r)
            continue
        normalized = normalize_fn(r["raw_response"])
        if normalized["parse_error"]:
            dropped += 1
            continue
        base = {k: r[k] for k in BASE_FIELDS}
        kept.append({**base, **normalized})
        recovered += 1

    atomic_write_json(path, kept)
    print(
        f"{path}: {len(records)} rows -> {len(kept)} kept "
        f"({recovered} recovered, {dropped} dropped for rerun)"
    )


def main():
    repair(EVAL_SCORES_DIR / f"{JUDGE_SLUG}.json", normalize_score_response)
    repair(EVAL_PASS_DIR / f"{JUDGE_SLUG}.json", normalize_pass_response)


if __name__ == "__main__":
    main()
