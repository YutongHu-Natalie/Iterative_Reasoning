"""AI-text detection with closed-source LLM judges (OpenAI / Anthropic APIs).

API counterpart of run_detection.py: same datasets (--data / --sample /
--paired-only / --limit), same prompt (prompt_template.txt), and the same
result-file format and summary, so API and open-source judges can be
compared directly. Models are listed in api_models.API_MODELS:
  gpt_6_luna_xhigh, gpt_6_1_sol_low, claude_haiku_5_5

--structure EXTRACTOR adds the structural annotation produced by
extract_structure.py --models EXTRACTOR to the prompt
(prompt_template_structure.txt) and writes <model_key>__struct-EXTRACTOR.json
next to the plain <model_key>.json, for compare_structure.py.

Each record also stores token usage and estimated cost; --max-cost caps the
spend of one invocation. Results are saved every --save-every items and
reruns resume; items that failed with an API error are retried, and
--retry-parse-errors also redoes unparseable answers.

Start small: try --limit 2 first to check the setup and the per-item cost.

Usage:
  python AI_Detect/Premiere/run_detection_api.py --data hard_balanced --models claude_haiku_5_5 --limit 2
  python AI_Detect/Premiere/run_detection_api.py --data hard_balanced --models all --max-cost 5
  python AI_Detect/Premiere/run_detection_api.py --data hard_balanced --models all --structure gpt_6_luna_xhigh
  python AI_Detect/Premiere/run_detection_api.py --data full --sample 200 --models gpt_6_1_sol_low
"""
import argparse
import time
from pathlib import Path

from api_models import (API_MODELS, Budget, add_api_args, call_model, model_config, print_cost,
                        run_pool)
from detect_common import (add_data_args, aggregate_runs, build_prompt, default_out_dir,
                           detection_template, item_meta, item_set_info, load_items, load_results,
                           load_structures, output_name, parse_verdict, save_results, summarize)

VERDICT_SCHEMA = {"name": "authorship_verdict", "schema": {
    "type": "object",
    "additionalProperties": False,
    "required": ["AI", "certainty"],
    "properties": {"AI": {"type": "integer", "enum": [0, 1]}, "certainty": {"type": "integer"}},
}}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_args(parser)
    add_api_args(parser)
    parser.add_argument("--structure", default=None, metavar="EXTRACTOR",
                        help="Also show the structural annotation from extract_structure.py --models EXTRACTOR.")
    parser.add_argument("--n-runs", type=int, default=1, help="Requests per item (majority vote).")
    parser.add_argument("--retry-parse-errors", action="store_true",
                        help="On resume, also redo items whose answers could not be parsed.")
    parser.add_argument("--out-dir", default=None, help="Defaults to TestResult/ (premiere) or TestResult/<data>/.")
    return parser.parse_args()


def main():
    args = parse_args()
    keys = list(API_MODELS) if "all" in args.models else list(dict.fromkeys(args.models))
    items = load_items(args.data, args.limit, args.sample, args.paired_only, args.seed)
    item_set = item_set_info(args, items)
    out_dir = Path(args.out_dir or default_out_dir(args.data))
    structures = load_structures(args.structure, items) if args.structure else None
    template = detection_template(structures is not None)
    schema = None if args.no_schema else VERDICT_SCHEMA
    budget = Budget(args.max_cost)
    print(f"{len(items)} items ({sum(i['is_ai'] for i in items)} LLM), models: {keys}"
          + (f", structure from {args.structure}" if args.structure else ""))

    def redo(record):
        return (any(run["api_error"] for run in record["runs"])
                or (args.retry_parse_errors and record["parse_error"]))

    for key in keys:
        start = time.time()
        out_path = out_dir / f"{output_name(key, args.structure)}.json"
        results = load_results(out_path, item_set, args.no_resume, drop=redo)
        done = {r["id"] for r in results}
        todo = [i for i in items if i["id"] not in done]
        if not todo:
            print(f"[{key}] already complete -> {out_path}")
            continue
        if done:
            print(f"[{key}] resuming: {len(done)} done, {len(todo)} to go")
        config = {**model_config(key), "structured_output": schema is not None,
                  "max_output_tokens": args.max_output_tokens, "n_runs": args.n_runs,
                  "structure_extractor": args.structure, "prompt_template": template}

        def work(item):
            prompt = build_prompt(template, item["problem"], item["solution"],
                                  structures[item["id"]] if structures else None)
            runs = []
            for run_idx in range(args.n_runs):
                out = call_model(key, prompt, schema, args.max_output_tokens)
                runs.append({
                    "run": run_idx,
                    **parse_verdict(out["text"]),
                    "status": out["status"],
                    "api_error": out["api_error"],
                    "hit_max_tokens": out["status"] == "max_tokens",
                    "usage": out["usage"],
                    "cost_usd": out["cost_usd"],
                    "raw_response": out["text"],
                    "thinking": out["thinking"],
                })
            return {**item_meta(item), **aggregate_runs(runs),
                    "cost_usd": sum(r["cost_usd"] for r in runs), "runs": runs}

        new = []

        def emit(record):
            results.append(record)
            new.append(record)
            if len(new) % args.save_every == 0:
                save_results(out_path, key, item_set, config, results, complete=False)

        try:
            left = run_pool(todo, work, emit, args.workers, budget, desc=key)
        finally:
            complete = len(results) == len(items) and not any(redo(r) for r in results)
            save_results(out_path, key, item_set, config, results, complete=complete)
        summary = summarize(results)
        print(f"[{key}] acc={summary['accuracy']} auroc={summary['auroc']} "
              f"parse_errors={summary['n_parse_errors']} -> {out_path} ({(time.time() - start) / 60:.1f} min)")
        print_cost(key, new)
        if left:
            print(f"Budget of ${args.max_cost} reached; {left} items of {key} not sent. Rerun to continue.")
            break


if __name__ == "__main__":
    main()
