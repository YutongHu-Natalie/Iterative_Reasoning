"""Annotate the visible mathematical structure of each solution with an API model.

Runs structure_prompt.txt (goal coverage, techniques, per-step claims /
roles / dependencies / validity / gaps, case structure, first invalid step,
...) on every item of --data and caches the JSON annotations in
Data/structure/<model_key>.json, keyed by item id. One cache per extractor
serves every dataset: hard-case subsets share ids with full, so annotating a
subset first and full later only pays for the new items.

The annotations are then fed to the detectors with --structure <model_key>
(run_detection_api.py for API models, run_detection.py for open-source
judges). The annotator never sees the authorship label.

Rerunning skips items that already have a valid annotation and retries
failed ones; --no-resume re-annotates this item set (other entries are kept).

Usage:
  python AI_Detect/Premiere/extract_structure.py --models gpt_6_luna_xhigh --data hard_balanced --limit 2
  python AI_Detect/Premiere/extract_structure.py --models all --data hard_balanced --max-cost 5
  python AI_Detect/Premiere/extract_structure.py --models gpt_6_1_sol_low --data full --sample 200
"""
import argparse
import hashlib
import json

from api_models import (API_MODELS, Budget, add_api_args, call_model, model_config, print_cost,
                        run_pool)
from detect_common import (HERE, add_data_args, atomic_write_json, build_prompt, extract_json,
                           load_items, structure_path)

PROMPT_PATH = HERE / "structure_prompt.txt"

_STEP = {
    "type": "object",
    "additionalProperties": False,
    "required": ["step_id", "claim", "role", "depends_on", "validity", "gap_severity",
                 "relevance", "error_type", "confidence"],
    "properties": {
        "step_id": {"type": "integer"},
        "claim": {"type": "string"},
        "role": {"type": "string", "enum": ["setup", "assumption", "calculation", "deduction",
                                            "theorem_application", "case", "verification",
                                            "conclusion", "other"]},
        "depends_on": {"type": "array", "items": {"type": "integer"}},
        "validity": {"type": "string", "enum": ["valid", "invalid", "uncertain"]},
        "gap_severity": {"type": "string", "enum": ["none", "minor", "major"]},
        "relevance": {"type": "string", "enum": ["necessary", "supportive", "irrelevant"]},
        "error_type": {"type": "string", "enum": ["none", "algebra", "arithmetic", "logic",
                                                  "theorem_misuse", "hidden_assumption",
                                                  "unsupported_generalization", "omitted_case",
                                                  "notation", "other"]},
        "confidence": {"type": "number"},
    },
}
STRUCTURE_SCHEMA = {"name": "solution_structure", "schema": {
    "type": "object",
    "additionalProperties": False,
    "required": ["goal_coverage", "techniques", "steps", "subgoal_count", "case_count",
                 "cases_exhaustive", "first_invalid_step", "error_affects_conclusion",
                 "verification_present", "conclusion_established"],
    "properties": {
        "goal_coverage": {"type": "string", "enum": ["complete", "partial", "incorrect"]},
        "techniques": {"type": "array", "items": {"type": "string", "enum": [
            "direct_proof", "contradiction", "induction", "casework", "construction", "invariant",
            "extremal", "counting", "algebra", "inequality", "modular_arithmetic", "parity",
            "geometry", "other"]}},
        "steps": {"type": "array", "items": _STEP},
        "subgoal_count": {"type": "integer"},
        "case_count": {"type": "integer"},
        "cases_exhaustive": {"type": "string", "enum": ["yes", "no", "uncertain", "not_applicable"]},
        "first_invalid_step": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
        "error_affects_conclusion": {"type": "boolean"},
        "verification_present": {"type": "boolean"},
        "conclusion_established": {"type": "string", "enum": ["complete", "partial", "not_established",
                                                              "uncertain"]},
    },
}}


def parse_annotation(text):
    parsed, error = extract_json(text or "")
    if parsed is None:
        return None, error
    if not isinstance(parsed, dict) or not isinstance(parsed.get("steps"), list):
        return None, f"missing steps: {str(parsed)[:200]}"
    return parsed, None


def load_cache(path, config):
    if not path.exists():
        return {"extractor": config["model_key"], "config": config, "annotations": {}}
    cache = json.load(open(path, encoding="utf-8"))
    if cache["config"]["prompt_sha1"] != config["prompt_sha1"]:
        raise SystemExit(f"{path} was made with a different structure_prompt.txt; "
                         "move it aside (or restore the prompt) before annotating more items.")
    return cache


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_args(parser)
    add_api_args(parser)
    args = parser.parse_args()

    keys = list(API_MODELS) if "all" in args.models else list(dict.fromkeys(args.models))
    items = load_items(args.data, args.limit, args.sample, args.paired_only, args.seed)
    template = PROMPT_PATH.read_text(encoding="utf-8")
    schema = None if args.no_schema else STRUCTURE_SCHEMA
    budget = Budget(args.max_cost)
    print(f"{len(items)} items, extractors: {keys}")

    for key in keys:
        path = structure_path(key)
        config = {"model_key": key, **model_config(key), "structured_output": schema is not None,
                  "max_output_tokens": args.max_output_tokens,
                  "prompt_sha1": hashlib.sha1(template.encode()).hexdigest(), "prompt_template": template}
        cache = load_cache(path, config)
        annotations = cache["annotations"]
        todo = [i for i in items if args.no_resume or (annotations.get(i["id"]) or {}).get("annotation") is None]
        if not todo:
            print(f"[{key}] all {len(items)} items already annotated -> {path}")
            continue
        print(f"[{key}] {len(items) - len(todo)} cached, {len(todo)} to annotate")

        def work(item):
            out = call_model(key, build_prompt(template, item["problem"], item["solution"]),
                             schema, args.max_output_tokens)
            annotation, error = parse_annotation(out["text"]) if out["text"] else (None, out["status"])
            return {"id": item["id"], "annotation": annotation, "parse_error": annotation is None,
                    "parse_msg": error, "status": out["status"], "api_error": out["api_error"],
                    "usage": out["usage"], "cost_usd": out["cost_usd"],
                    "raw_response": out["text"], "thinking": out["thinking"]}

        new = []

        def emit(record):
            new.append(record)
            annotations[record.pop("id")] = record
            if len(new) % args.save_every == 0:
                atomic_write_json(path, cache)

        try:
            left = run_pool(todo, work, emit, args.workers, budget, desc=key)
        finally:
            atomic_write_json(path, cache)
        n_bad = sum(r["parse_error"] for r in new)
        print(f"[{key}] annotated {len(new) - n_bad}/{len(new)} ok -> {path}")
        print_cost(key, new)
        if left:
            print(f"Budget of ${args.max_cost} reached; {left} items not sent. Rerun to continue.")
            break


if __name__ == "__main__":
    main()
