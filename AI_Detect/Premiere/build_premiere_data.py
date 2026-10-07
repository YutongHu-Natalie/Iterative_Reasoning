"""Extract paired LLM / human proofs for the AI-detection premiere test.

For each selected problem, take the first non-empty LLM solution encountered and the
human-written reference solution.
"""
import json
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
DATA_DIR = os.path.join(ROOT, "MathProofDatasets")
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data", "premiere_data.json")

MATHARENA = {
    "usamo_2026": ["3", "5", "6"],
    "usamo_2025": ["1", "2", "4", "6"],
}
OPC_IDS = [
    "BMOSL_2016_18", "BMOSL_2018_15", "BMOSL_2020_13", "BMOSL_2018_16",
    "BMOSL_2021_7", "USAMO_2013_4", "USAMO_2015_5", "IMOSL_2010_1",
    "BMOSL_2016_2", "BMOSL_2017_23", "BMOSL_2018_18", "BMOSL_2017_17",
    "BMOSL_2017_18",
]
OPC_FILE = "best_of_n-00000-of-00001.json"


def load(path):
    with open(path) as f:
        return json.load(f)


records = []

for comp, idxs in MATHARENA.items():
    problems = {str(p["problem_idx"]): p for p in load(os.path.join(DATA_DIR, "MathArena", comp, "data", "train-00000-of-00001.json"))}
    outputs = load(os.path.join(DATA_DIR, "MathArena", f"{comp}_outputs", "data", "train-00000-of-00001.json"))
    for idx in idxs:
        first = next(o for o in outputs if str(o["problem_idx"]) == idx and o["answer"])
        records.append({
            "id": f"matharena_{comp}_{idx}",
            "source": "MathArena",
            "problem": problems[idx]["problem"],
            "llm_solution": first["answer"],
            "llm_model": first["model_name"],
            "human_solution": problems[idx]["sample_solution"],
        })

opc = load(os.path.join(DATA_DIR, "OPC", "data", OPC_FILE))
for pid in OPC_IDS:
    first = next(r for r in opc if r["problem_id"] == pid and r["solution"])
    records.append({
        "id": f"opc_{pid}",
        "source": "OPC",
        "problem": first["problem"],
        "llm_solution": first["solution"],
        "llm_model": first["model_id"],
        "human_solution": first["ground_truth_solution"],
    })

for r in records:
    for k in ("llm_solution", "human_solution"):
        if not r[k] or r[k] == "None":
            print(f"WARNING: empty {k} for {r['id']}")

with open(OUT_PATH, "w") as f:
    json.dump(records, f, indent=2, ensure_ascii=False)
print(f"Wrote {len(records)} records to {OUT_PATH}")
