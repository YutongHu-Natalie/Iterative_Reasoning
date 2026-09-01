"""Repair pass for every Results/eval_scores/*.json and Results/eval_pass/*.json
judge output file.

Every judge model here answers in a chat-completion style that's prone to two
kinds of near-miss JSON, both handled by eval_utils.extract_json():

  - math-tuned models (Qwen2.5-Math) answering inside a \\boxed{}/\\text{}
    LaTeX wrapper instead of plain JSON
  - otherwise well-formed JSON with unescaped LaTeX backslashes inside string
    values (e.g. "...the vector \\mathbf{w}...", which \\m isn't a legal JSON
    escape for)

This script re-normalizes every already-collected raw_response against the
current extract_json() so already-collected data can be salvaged without
rerunning the model:

  - rows that now parse are rewritten with their recovered fields (matching
    normalize_score_response/normalize_pass_response's own schema exactly, so
    this is not a new code path, just applying the current one retroactively)
  - rows that still fail to parse are dropped, not left in place, so that
    rerunning the relevant run_<model>_judge.py script (resumable by
    (judged_run, id)) regenerates exactly those rows instead of silently
    skipping them as already "done"

A dropped row with an EMPTY raw_response (as opposed to malformed-but-present
text) has no text to repair -- that means the judge script itself discarded
its output before ever writing it (see run_gemma4_e4b_judge.py's thinking-mode
fallback). No text repair can recover those; they're reported separately
below so it's clear a source-side fix + rerun is needed, not just a rerun.

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

BASE_FIELDS = (
    "id", "domain", "difficulty", "truth_value", "source",
    "judged_run", "judged_model", "judged_thinking_mode", "judge_model",
)


def repair(path, normalize_fn):
    records = load_existing(path)
    if not records:
        return
    kept = []
    recovered = 0
    dropped_with_text = 0
    dropped_empty = 0
    for r in records:
        if not r.get("parse_error"):
            kept.append(r)
            continue
        raw = r.get("raw_response") or ""
        normalized = normalize_fn(raw)
        if normalized["parse_error"]:
            if raw.strip():
                dropped_with_text += 1
            else:
                dropped_empty += 1
            continue
        base = {k: r[k] for k in BASE_FIELDS}
        kept.append({**base, **normalized})
        recovered += 1

    atomic_write_json(path, kept)
    msg = (
        f"{path}: {len(records)} rows -> {len(kept)} kept "
        f"({recovered} recovered, {dropped_with_text} dropped for rerun"
    )
    if dropped_empty:
        msg += f", {dropped_empty} dropped with EMPTY raw_response -- needs a source-side fix, not just a rerun"
    print(msg + ")")


def main():
    for path in sorted(EVAL_SCORES_DIR.glob("*.json")):
        repair(path, normalize_score_response)
    for path in sorted(EVAL_PASS_DIR.glob("*.json")):
        repair(path, normalize_pass_response)


if __name__ == "__main__":
    main()
