"""Build the full AI-vs-human proof detection dataset.

Sources (in priority order -- a problem is attributed to the first source it
appears in):
  1. MathArena  usamo_2025, usamo_2026, imc_2025, miklos_2025, putnam_2025
                LLM: model outputs with full marks; human: sample_solution
  2. OPC        all splits; LLM: rows every judge scored 1; human: ground truth
  3. ProofBench LLM: expert_rating == 7; human: reference_solution
  4. HARP       human only (all AoPS solutions); NEW problems only
  5. OlympiadBench  human only; NEW problems only

Rules:
  * Only theorem-proving problems: the statement must contain a proof keyword.
  * LLM proofs must be graded fully correct; keep the first one per
    (problem, generator) so heavily-sampled generators don't dominate.
  * Human reference solutions: one source per problem (HARP / OlympiadBench
    contribute all their solutions, but only for problems not seen before).
  * Cross-source duplicates are found by (contest, year, number) keys and by
    statement similarity (8-char shingle containment, only against OTHER
    datasets -- ids are already unique within a dataset). Borderline pairs
    count as duplicates for HARP / OlympiadBench (new problems only) and as
    distinct problems elsewhere; all are written to
    Data/full_data_dedup_review.json for manual checking.
  * Giveaway boilerplate is stripped from both sides (see clean_solution);
    every record lists the cleaning steps that fired.
  * Categories use one scheme: Algebra / Combinatorics / Geometry / Number
    Theory / Analysis. Source labels are mapped (CATEGORY_MAP); a problem with
    no label borrows one from a duplicate copy in a later source; the rest,
    plus ambiguous or multi-label cases, are filled from
    Data/category_llm_labels.json (written by label_categories.py) and are
    marked category_source="pending_llm" until that file exists.

Outputs (Data/):
  full_data.json               one record per solution
  full_data_dedup_review.json  borderline duplicate pairs
  full_data_build_log.json     counts per source / filter step
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE.parent.parent / "MathProofDatasets"
OUT_DIR = HERE / "Data"

PROOF_KW = re.compile(r"\b(prove|proof|show|demonstrate|establish|verify|disprove)\b", re.I)

MATHARENA_COMPS = {
    "usamo_2025": ("USAMO", 2025),
    "usamo_2026": ("USAMO", 2026),
    "imc_2025": ("IMC", 2025),
    "miklos_2025": ("Miklos", 2025),
    "putnam_2025": ("Putnam", 2025),
}

# Same model under different names across datasets -> one canonical name.
GENERATOR_ALIASES = {
    "openai/o3": "o3", "OpenAI-o3": "o3",
    "openai/o4-mini--high": "o4-mini (high)", "o4-mini (high)": "o4-mini (high)",
    "gemini/gemini-pro-2.5": "Gemini 2.5 Pro", "Gemini-2.5-pro": "Gemini 2.5 Pro",
    "Gemini 2.5 Pro": "Gemini 2.5 Pro",
    "deepseek/r1_0528": "DeepSeek-R1-0528", "DeepSeek-R1-0528": "DeepSeek-R1-0528",
    "qwen/qwen3_235b_a22b": "Qwen3-235B-A22B",
    "xai/grok-3-mini-high": "Grok 3 Mini (high)",
}

# Statement-similarity thresholds (8-char shingle containment). Calibrated by
# inspection: >= 0.70 is almost always the same problem (often re-typeset);
# 0.35-0.70 is mostly different problems with shared phrasing, plus a few
# rephrasings, so it is logged for review.
DUP_AUTO = 0.70
DUP_BORDER = 0.35
MIN_SOLUTION_CHARS = 80

CATEGORIES = ("Algebra", "Combinatorics", "Geometry", "Number Theory", "Analysis")
# Source label -> standard category. HARP labels are MATH-dataset subjects;
# OPC Putnam labels are PutnamBench tags.
CATEGORY_MAP = {
    "algebra": "Algebra", "linear algebra": "Algebra", "abstract algebra": "Algebra",
    "geometry": "Geometry",
    "number theory": "Number Theory",
    "combinatorics": "Combinatorics", "counting and probability": "Combinatorics",
    "analysis": "Analysis",
}
# Labels that don't determine one category: the LLM picks among these options.
CATEGORY_AMBIGUOUS = {
    "precalculus": ("Algebra", "Geometry"),   # trig / complex numbers / vectors
    "set theory": CATEGORIES,
}
CATEGORY_LLM_LABELS = HERE / "Data" / "category_llm_labels.json"

REFUSAL = re.compile(
    r"(unable to (solve|find|prove)|i('m| am) sorry|could not (find|solve|prove)|"
    r"cannot provide a (complete|full)|no (complete )?proof (was|could be) found)", re.I)


# ---------------------------------------------------------------- helpers

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def to_int(x):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


def norm_category(cat):
    if cat is None:
        return None
    if isinstance(cat, list):
        parts = [norm_category(c) for c in cat]
        parts = [p for p in parts if p]
        return "; ".join(parts) or None
    cat = str(cat).strip().replace("_", " ").lower()
    return cat or None


def canonical_generator(raw):
    base = raw.split(" (")[0] if raw.startswith(("openai/", "gemini/", "deepseek/", "qwen/", "xai/")) else raw
    return GENERATOR_ALIASES.get(base, GENERATOR_ALIASES.get(raw, raw))


def contest_key(contest, year, number):
    if contest is None or year is None or number is None:
        return None
    contest = {"USAJMO": "JMO"}.get(contest.upper(), contest.upper())
    return f"{contest}|{year}|{str(number).lower()}"


def statement_tokens(text):
    text = re.sub(r"\\[a-zA-Z]+", " ", text.lower())
    return set(re.findall(r"[a-z]{2,}|\d+", text))


def statement_shingles(text, k=8):
    """Character k-grams of the statement with LaTeX markup and spacing removed,
    so '$A B C$' and '\\( ABC \\)' compare equal."""
    text = re.sub(r"\\(left|right|mathrm|mathbb|mathcal|text|operatorname|displaystyle|quad|qquad|"
                  r"cdot|ldots|cdots|dots)\b", "", text.lower())
    text = re.sub(r"\\[a-z]+", "", text)
    text = re.sub(r"[^a-z0-9]", "", text)
    return {text[i:i + k] for i in range(max(1, len(text) - k + 1))}


def jaccard(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


# ---------------------------------------------------------------- cleaning

QED_TAIL = re.compile(
    r"\s*(\$\s*\\(blacksquare|square|Box|qed)\s*\$|\\\(\s*\\(blacksquare|square|Box)\s*\\\)|"
    r"\\(blacksquare|square|Box|qed)\b|[∎■□]|Q\.?\s?E\.?\s?D\.?)\s*$")
HEADING_LEAD = re.compile(
    r"^\s*(#{1,6}[^\n]*\n|\\section\*?\{[^}]*\}|\\text\{noindent\}|\\noindent|"
    r"<strong>[^<]{0,40}</strong>\s*[.:]?|\\textbf\{\s*(proof|solution)[^}]{0,30}\}\s*[.:]?|\*\*\s*(proof|solution)[^*\n]{0,30}\*\*\s*[.:]?|"
    r"(proof|solution)(\s+\d+(\.\d+)?)?\s*[.:](?!\d))\s*", re.I)
LLM_INTRO = re.compile(
    r"^\s*(here is|here's|below is|the following is)\b[^.:]{0,40}?\b(proof|solution|argument|write-?up|construction|"
    r"approach|derivation)\b[\s\S]{0,400}?[.:](\s|$)", re.I)
TASK_RESTATE = re.compile(r"^\s*(the problem (requires|asks)|the goal is|we are asked)\b[^\n]*?\.(\s|$)", re.I)
TO_X_LEAD = re.compile(
    r"^\s*to (solve|prove|show|find|determine|address)\b[^\n]{0,300}?,\s+(?=(we|let|i|first|consider|note|start)\b)",
    re.I)
# Self-description that follows "Here is a self-contained proof." in o4-mini/o3 outputs.
LLM_SELF_DESCRIPTION = re.compile(
    r"^\s*((all|every) (steps?|details?|claims?)[^.\n]{0,40}\b(justified|verified|explained|included|written|elementary)|"
    r"we (write|spell) (out )?(everything|all (the )?details|every step)|"
    r"no (high-level|advanced|deep)\b|(the|this) (proof|argument) is (fully )?(self[-‐–]contained|elementary))"
    r"[^\n]*?\.(\s|$)", re.I)
# A whole final line of self-assessment (Gemini / o4-mini habit), incl. leaked
# reasoning like "Final answer format: Need LaTeX.".
LLM_META_TAIL = re.compile(
    r"\n?[ \t]*((all|every) steps? (are|seem|have been)\b[^\n]*\b(justified|verified|correct|shown|included)|"
    r"(the|this) (proof|argument|solution) (is|was) (now )?(complete(ly)?|fully|rigorous|self[-‐–]contained)\b|"
    r"the argument relies on|final answer format|correctness relies)[^\n]*$", re.I)
# Qwen-style closing section: "---\n### Final Answer\n$$ <restated result> $$".
FINAL_ANSWER_SECTION = re.compile(
    r"\n([ \t]*-{3,}[ \t]*\n)?\s*(#{1,6}\s*)?(\*\*)?\s*(\d+\.\s*)?final answers?\s*:?\s*(\*\*)?[ \t]*\n[\s\S]{0,400}$",
    re.I)
# Self-assessment sentences anywhere in LLM text.
# ("The final answer is ..." runs to the end of its line: the restated result
# is usually math containing periods.)
LLM_META_SENTENCE = re.compile(
    r"(?:(?<=[.!?:])|^)[ \t]*((all|every) steps? (are|have been|seem to be) (fully )?(correct and )?(justified|verified)"
    r"[^\n$]*?\.(?=\s|$)|the final answer is\b[^\n]*)", re.I | re.M)
FINAL_ANSWER_HEADING = re.compile(r"^[ \t]*(#{1,6}\s*)?(\*\*)?\s*(\d+\.\s*)?final answers?\s*:?\s*(\*\*)?[ \t]*\n?",
                                  re.I | re.M)
PLEASANTRY = re.compile(r"^\s*(sure|certainly|of course|great question)\b[^\n]*\n", re.I)
FINAL_ANSWER_TAIL = re.compile(r"\n?[^\n]*\b(the )?final answer\b[^\n]*$", re.I)
BOILERPLATE_BOX = re.compile(
    r"\s*(The final answer is\s*)?(\$\$?|\\\[|\\\()?\s*\\boxed\{\s*\\text\{[^{}]*(proof|q\.?e\.?d|done|complete)[^{}]*\}\s*\}"
    r"\s*(\$\$?|\\\]|\\\))?\s*\.?\s*$", re.I)


def _match_brace(text, open_idx):
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    return None


def unbox(text):
    """\\boxed{X} -> X everywhere (keeps the answer, drops the LLM habit)."""
    out, i, changed = [], 0, False
    while True:
        j = text.find("\\boxed{", i)
        if j < 0:
            out.append(text[i:])
            break
        close = _match_brace(text, j + len("\\boxed"))
        if close is None:
            out.append(text[i:])
            break
        out.append(text[i:j])
        out.append(text[j + len("\\boxed{"):close])
        i, changed = close + 1, True
    return "".join(out), changed


def strip_trailing_box(text):
    """A final standalone \\boxed{...} restating the result (after a complete
    sentence) is dropped; a box that completes a sentence ('... is \\boxed{18}.')
    is left for unbox() so the answer survives."""
    j = text.rfind("\\boxed{")
    if j < 0:
        return text
    close = _match_brace(text, j + len("\\boxed"))
    if close is None or not re.fullmatch(r"[\s$.\\\])*]*", text[close + 1:]):
        return text
    head = re.sub(r"(\$\$|\$|\\\[|\\\(|\*\*)\s*$", "", text[:j]).rstrip()
    head_wo_label = re.sub(r"(\*\*)?\b(the )?(final )?answer( is)?\s*:?\s*(\*\*)?$", "", head, flags=re.I).rstrip()
    if head_wo_label != head or head.endswith((".", "]", ")", "$")) or head.endswith("\n"):
        return head_wo_label
    return text


def strip_problem_statement(text, problem):
    """OPC ground truths sometimes start with '## Problem statement' + the problem."""
    body = re.sub(r"^\s*##\s*Problem statement\s*\n", "", text)
    paras = re.split(r"\n\s*\n", body)
    target = statement_tokens(problem)
    best_k, best_sim = 0, 0.0
    for k in range(1, min(6, len(paras))):
        sim = jaccard(statement_tokens("\n".join(paras[:k])), target)
        if sim > best_sim:
            best_k, best_sim = k, sim
    if best_sim >= 0.6:
        return "\n\n".join(paras[best_k:])
    return body


def clean_solution(text, problem, author, dataset):
    """Strip authorship giveaways. Returns (clean_text, steps_applied)."""
    steps = []

    def apply(name, new):
        nonlocal text
        if new != text:
            steps.append(name)
            text = new

    text = text.replace("\r\n", "\n").strip()

    # --- source-specific
    if author == "human" and text.lstrip().startswith("## Problem statement"):
        apply("problem_statement", strip_problem_statement(text, problem))
    m = re.search(r"#+\s*Detailed Solution\s*#*", text)
    if author == "llm" and m:
        apply("agent_summary", text[m.end():])
    if author == "human" and dataset == "OPC":
        apply("compendium_number", re.sub(r"^\s*\d{1,2}\.\s+(?=[A-Z])", "", text))
    if author == "human" and dataset == "HARP":
        apply("aops_note", re.sub(r"^\s*Note:[^\n]*\n+", "", text))
        apply("aops_signature", re.sub(r"\s*~\s*[\w.\-]+(\s*(,|and)\s*~?\s*[\w.\-]+)*\s*$", "", text))
        apply("aops_signature", re.sub(r"(?<=[.!?$])\s+[—–-]\s*[A-Za-z][\w.\-]{2,30}\s*$", "", text))

    # --- markup / figure artifacts (anywhere)
    apply("asy_figure", re.sub(r"\[asy\].*?\[/asy\]", "", text, flags=re.S))
    apply("image_placeholder", re.sub(r"<img_\d+>", "", text))
    apply("latex_figure", re.sub(r"\\begin\{figure\}.*?\\end\{figure\}", "", text, flags=re.S))
    apply("latex_figure", re.sub(r"\\includegraphics(\[[^\]]*\])?\{[^}]*\}", "", text))
    apply("html_tags", re.sub(r"</?(strong|em|ul|ol|li|p|br|b|i)\s*/?>", "", text))
    apply("section_heading", re.sub(r"^\s*\\section\*?\{[^}]*\}\s*$", "", text, flags=re.M))

    apply("mojibake", text.replace("гЂђ", "").replace("гЂ‘", ""))

    # --- leading boilerplate (repeat: e.g. heading then "Here is ...").
    # LLM-phrasing patterns run on LLM text only: on human text they only
    # ever produced false positives.
    is_llm = author == "llm"
    for _ in range(4):
        before = text
        apply("heading", HEADING_LEAD.sub("", text, count=1))
        if is_llm:
            apply("pleasantry", PLEASANTRY.sub("", text, count=1))
            apply("llm_intro", LLM_INTRO.sub("", text, count=1))
            apply("llm_self_description", LLM_SELF_DESCRIPTION.sub("", text, count=1))
            apply("task_restatement", TASK_RESTATE.sub("", text, count=1))
            apply("to_x_lead", TO_X_LEAD.sub("", text, count=1))
        if text == before:
            break
    if text and text[0].islower() and "to_x_lead" in steps:
        text = text[0].upper() + text[1:]

    if is_llm:
        apply("final_answer_section", FINAL_ANSWER_SECTION.sub("", text))
        apply("final_answer_heading", FINAL_ANSWER_HEADING.sub("", text))
        # Skip a match whose braces don't balance (a box spanning several lines).
        apply("llm_meta_sentence", LLM_META_SENTENCE.sub(
            lambda m: "" if m.group(0).count("{") == m.group(0).count("}") else m.group(0), text))

    # --- trailing boilerplate, peeled until stable (e.g. QED mark, then a
    # boxed restatement, then a self-assessment line)
    for _ in range(5):
        before = text
        apply("qed_marker", QED_TAIL.sub("", text).rstrip())
        if is_llm:
            apply("boilerplate_box", BOILERPLATE_BOX.sub("", text))
            apply("final_answer_line", FINAL_ANSWER_TAIL.sub("", text) if len(text) > 400 else text)
            apply("llm_meta_tail", LLM_META_TAIL.sub("", text).rstrip())
        apply("trailing_box", strip_trailing_box(text))
        apply("trailing_rule", re.sub(r"\n[ \t]*-{3,}[ \t]*$", "", text))
        if text == before:
            break
    apply("unbox", unbox(text)[0])

    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text, sorted(set(steps), key=steps.index)


# ---------------------------------------------------------------- dedup registry

class ProblemRegistry:
    def __init__(self):
        self.entries = []                  # {puid, dataset, shingles, problem}
        self.index = defaultdict(set)      # shingle -> entry indices
        self.by_key = {}
        self.review = []

    def best_match(self, problem, dataset):
        """Most similar registered problem from a different dataset."""
        sh = statement_shingles(problem)
        hits = Counter(j for g in sh for j in self.index[g] if self.entries[j]["dataset"] != dataset)
        best, best_sim = None, 0.0
        for j, n in hits.items():
            sim = n / min(len(sh), len(self.entries[j]["shingles"]))
            if sim > best_sim:
                best, best_sim = self.entries[j], sim
        return best, best_sim

    def match(self, problem, key, dataset, ref, borderline_is_duplicate):
        """Return (puid or None, kind) where kind in {key, auto, borderline, new}."""
        if key and key in self.by_key:
            return self.by_key[key], "key"
        best, sim = self.best_match(problem, dataset)
        if best and sim >= DUP_AUTO:
            return best["puid"], "auto"
        if best and sim >= DUP_BORDER:
            self.review.append({
                "similarity": round(sim, 3),
                "treated_as_duplicate": borderline_is_duplicate,
                "new": {"dataset": dataset, "ref": ref, "problem": problem},
                "existing": {"dataset": best["dataset"], "puid": best["puid"], "problem": best["problem"]},
            })
            return (best["puid"], "borderline") if borderline_is_duplicate else (None, "borderline_kept")
        return None, "new"

    def add(self, puid, problem, key, dataset):
        j = len(self.entries)
        sh = statement_shingles(problem)
        self.entries.append({"puid": puid, "dataset": dataset, "shingles": sh, "problem": problem})
        for g in sh:
            self.index[g].add(j)
        if key:
            self.by_key[key] = puid


# ---------------------------------------------------------------- builder

class Builder:
    def __init__(self):
        self.registry = ProblemRegistry()
        self.records = []
        self.problem_meta = {}             # puid -> meta of first source
        self.has_human = set()             # puids with a human solution
        self.llm_seen = set()              # (puid, generator_variant)
        self.log = defaultdict(Counter)

    def resolve_problem(self, dataset, ref, problem, key, meta, new_only=False):
        """Map a source problem to a canonical puid; None if it should be skipped."""
        puid, kind = self.registry.match(problem, key, dataset, ref, borderline_is_duplicate=new_only)
        self.log[dataset][f"problem_match_{kind}"] += 1
        if puid is None:
            puid = f"{dataset}:{ref}"
            self.registry.add(puid, problem, key, dataset)
            self.problem_meta[puid] = {**meta, "problem": problem, "dataset": dataset, "problem_idx": ref}
            return puid, True
        if key and key not in self.registry.by_key:
            self.registry.by_key[key] = puid
        # Borrow a category from this duplicate copy if the canonical copy has none.
        if kind != "borderline" and meta.get("category") and not self.problem_meta[puid].get("category"):
            self.problem_meta[puid]["category"] = meta["category"]
            self.problem_meta[puid]["category_from"] = dataset
        return (None, False) if new_only else (puid, False)

    def add_solution(self, puid, dataset, author, raw_solution, generator=None, source_meta=None):
        log = self.log[dataset]
        if not raw_solution or not str(raw_solution).strip():
            log["drop_empty"] += 1
            return
        if author == "llm":
            variant = generator
            if (puid, variant) in self.llm_seen:
                log["drop_llm_repeat_problem_model"] += 1
                return
            if REFUSAL.search(raw_solution[:400]):
                log["drop_refusal"] += 1
                return
        meta = self.problem_meta[puid]
        solution, steps = clean_solution(raw_solution, meta["problem"], author, dataset)
        if len(solution) < MIN_SOLUTION_CHARS:
            log["drop_too_short_after_cleaning"] += 1
            return
        if author == "llm":
            self.llm_seen.add((puid, generator))
        else:
            self.has_human.add(puid)
        n = sum(1 for r in self.records if r["problem_uid"] == puid and r["author"] == author
                and r["generator_variant"] == generator)
        self.records.append({
            "uid": f"{puid}|{author}|{generator or 'human'}|{n}",
            "problem_uid": puid,
            "dataset": dataset,
            "competition": meta.get("competition"),
            "year": meta.get("year"),
            "problem_idx": meta.get("problem_idx"),
            "category": meta.get("category"),
            "problem": meta["problem"],
            "solution": solution,
            "author": author,
            "is_ai": int(author == "llm"),
            "generator_model": canonical_generator(generator) if generator else None,
            "generator_variant": generator,
            "source_meta": source_meta or {},
            "cleaning": steps,
        })
        log[f"kept_{author}"] += 1

    # ---------------- sources

    def matharena(self):
        for comp, (contest, year) in MATHARENA_COMPS.items():
            base = DATA_ROOT / "MathArena"
            problems = {str(p["problem_idx"]): p for p in load_json(base / comp / "data" / "train-00000-of-00001.json")}
            outputs = load_json(base / f"{comp}_outputs" / "data" / "train-00000-of-00001.json")
            texts = {str(o["problem_idx"]): o["problem"] for o in outputs}
            texts.update({i: p["problem"] for i, p in problems.items()})
            puids = {}
            for idx in sorted(texts, key=lambda x: int(x)):
                if not PROOF_KW.search(texts[idx]):
                    self.log["MathArena"]["drop_not_proof_problem"] += 1
                    continue
                key = contest_key(contest, year, idx) if contest != "Putnam" else None
                puid, _ = self.resolve_problem("MathArena", f"{comp}_{idx}", texts[idx], key,
                                               {"competition": contest, "year": year, "category": None})
                puids[idx] = puid
                human = problems.get(idx, {}).get("sample_solution")
                if human and puid not in self.has_human:
                    self.add_solution(puid, "MathArena", "human", human, source_meta={"source": comp})
            for o in sorted(outputs, key=lambda o: (int(o["problem_idx"]), o.get("idx_answer", 0))):
                idx = str(o["problem_idx"])
                if idx not in puids:
                    continue
                if o.get("correct") != 1:
                    self.log["MathArena"]["drop_llm_not_full_marks"] += 1
                    continue
                self.add_solution(puids[idx], "MathArena", "llm", o["answer"], o["model_name"],
                                  {"source": comp, "idx_answer": o.get("idx_answer"), "correct": o["correct"]})

    def opc(self):
        files = sorted((DATA_ROOT / "OPC" / "data").glob("*.json"))
        rows = [(f.name.split("-")[0], r) for f in files for r in load_json(f)]
        puids = {}
        for split, r in rows:
            pid = r["problem_id"]
            if pid not in puids:
                if not PROOF_KW.search(r["problem"]):
                    puids[pid] = None
                    self.log["OPC"]["drop_not_proof_problem"] += 1
                    continue
                m = re.match(r"^([A-Za-z0-9-]+)_(\d{4})_([A-Za-z0-9]+)$", pid)
                year = to_int(r.get("year")) or (int(m.group(2)) if m else None)
                key = contest_key(m.group(1), int(m.group(2)), m.group(3)) if m else None
                puid, _ = self.resolve_problem("OPC", pid, r["problem"], key, {
                    "competition": r.get("competition"), "year": year, "category": norm_category(r.get("category"))})
                puids[pid] = puid
                gt = r.get("ground_truth_solution")
                if gt and gt != "None" and puid not in self.has_human:
                    self.add_solution(puid, "OPC", "human", gt, source_meta={"opc_split": split})
            puid = puids[pid]
            if puid is None:
                continue
            scores = r.get("score") or []
            if not scores or not all(s == 1 for s in scores):
                self.log["OPC"]["drop_llm_not_all_judges_correct"] += 1
                continue
            model_id = r["model_id"]
            selector = model_id[len(model_id.split(" (")[0]):].strip() or None
            self.add_solution(puid, "OPC", "llm", r["solution"], model_id.split(" (")[0],
                              {"opc_split": split, "opc_selector": selector, "score": scores})

    def proofbench(self):
        files = sorted((DATA_ROOT / "ProofBench" / "data").glob("*.json"), key=lambda p: p.name != "train-00000-of-00001.json")
        puids = {}
        for f in files:
            for r in load_json(f):
                pid = r["problem_id"]
                meta = r["metadata"] if isinstance(r["metadata"], dict) else {}
                if pid not in puids:
                    if not PROOF_KW.search(r["problem"]):
                        puids[pid] = None
                        self.log["ProofBench"]["drop_not_proof_problem"] += 1
                        continue
                    m = re.match(r"^(.+)-(\d{4})-([A-Za-z0-9]+)$", pid)
                    key = contest_key(m.group(1), int(m.group(2)), m.group(3)) if m else None
                    puid, _ = self.resolve_problem("ProofBench", pid, r["problem"], key, {
                        "competition": meta.get("contest"), "year": to_int(meta.get("contest_year")), "category": None})
                    puids[pid] = puid
                    if puid not in self.has_human:
                        self.add_solution(puid, "ProofBench", "human", r.get("reference_solution"),
                                          source_meta={"split": f.name.split("-")[0]})
                puid = puids[pid]
                if puid is None:
                    continue
                if r.get("expert_rating") != 7:
                    self.log["ProofBench"]["drop_llm_not_full_marks"] += 1
                    continue
                self.add_solution(puid, "ProofBench", "llm", r["model_solution"], r["generator"],
                                  {"split": f.name.split("-")[0], "expert_rating": r["expert_rating"],
                                   "response_number": r.get("response_number")})

    def harp(self):
        with open(DATA_ROOT / "HARP_prove_proof_only.jsonl", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        for r in rows:
            if not PROOF_KW.search(r["problem"]):
                self.log["HARP"]["drop_not_proof_problem"] += 1
                continue
            ref = f"{r['contest']}_{r['year']}_{r['number']}"
            puid, is_new = self.resolve_problem("HARP", ref, r["problem"],
                                                contest_key(r["contest"], to_int(r["year"]), r["number"]),
                                                {"competition": r["contest"], "year": to_int(r["year"]),
                                                 "category": norm_category(r.get("subject"))}, new_only=True)
            if not is_new:
                self.log["HARP"]["drop_problem_seen_before"] += 1
                continue
            for i in range(1, (to_int(r.get("num_solutions")) or 0) + 1):
                self.add_solution(puid, "HARP", "human", r.get(f"solution_{i}"), source_meta={"solution_number": i})

    def olympiadbench(self):
        for r in load_json(DATA_ROOT / "OlympiadBench_TP_TO_maths_en_COMP.json"):
            problem = r["question"] if not r.get("context") or r["context"] == "None" else f"{r['context']}\n\n{r['question']}"
            if not PROOF_KW.search(problem):
                self.log["OlympiadBench"]["drop_not_proof_problem"] += 1
                continue
            puid, is_new = self.resolve_problem("OlympiadBench", str(r["id"]), problem, None, {
                "competition": None, "year": None, "category": norm_category(r.get("subfield"))}, new_only=True)
            if not is_new:
                self.log["OlympiadBench"]["drop_problem_seen_before"] += 1
                continue
            sols = r["solution"] if isinstance(r["solution"], list) else [r["solution"]]
            for i, s in enumerate(sols, 1):
                self.add_solution(puid, "OlympiadBench", "human", s, source_meta={"solution_number": i})

    def assign_categories(self):
        """Map raw labels onto CATEGORIES; fill the rest from the LLM label file."""
        llm_labels = load_json(CATEGORY_LLM_LABELS) if CATEGORY_LLM_LABELS.exists() else {}
        resolved = {}
        for puid, meta in self.problem_meta.items():
            raw = meta.get("category")
            tags, options = [], []
            for label in (raw.split("; ") if raw else []):
                if label in CATEGORY_MAP:
                    tags.append(CATEGORY_MAP[label])
                options += CATEGORY_AMBIGUOUS.get(label, ())
            tags = list(dict.fromkeys(tags))
            if len(tags) == 1 and not options:
                category, source = tags[0], "source"
            else:
                allowed = list(dict.fromkeys(tags + list(options))) or list(CATEGORIES)
                label = llm_labels.get(puid, {}).get("category")
                if label in allowed:
                    category, source = label, ("llm" if not tags else "llm_tiebreak")
                else:
                    category, source = None, "pending_llm"
                meta["category_allowed"] = allowed
            resolved[puid] = {
                "category": category,
                "category_tags": tags or ([category] if category else []),
                "category_raw": raw,
                "category_source": source,
                "category_raw_from": meta.get("category_from", meta["dataset"]) if raw else None,
            }
        for r in self.records:
            r.update(resolved[r["problem_uid"]])
        self.category_pending = {
            puid: {"problem": m["problem"], "competition": m.get("competition"),
                   "category_raw": m.get("category"), "allowed": m["category_allowed"]}
            for puid, m in self.problem_meta.items()
            if resolved[puid]["category_source"] == "pending_llm"
            and any(r["problem_uid"] == puid for r in self.records)
        }

    def drop_exact_duplicates(self):
        seen, kept = set(), []
        for r in self.records:
            key = re.sub(r"\s+", " ", r["solution"]).strip()
            if key in seen:
                self.log[r["dataset"]]["drop_exact_duplicate_text"] += 1
                self.log[r["dataset"]][f"kept_{r['author']}"] -= 1
                continue
            seen.add(key)
            kept.append(r)
        self.records = kept


def main():
    b = Builder()
    b.matharena()
    b.opc()
    b.proofbench()
    b.harp()
    b.olympiadbench()
    b.drop_exact_duplicates()
    b.assign_categories()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "category_pending.json", "w", encoding="utf-8") as f:
        json.dump(b.category_pending, f, indent=2, ensure_ascii=False)
    with open(OUT_DIR / "full_data.json", "w", encoding="utf-8") as f:
        json.dump(b.records, f, indent=2, ensure_ascii=False)
    with open(OUT_DIR / "full_data_dedup_review.json", "w", encoding="utf-8") as f:
        json.dump(sorted(b.registry.review, key=lambda x: -x["similarity"]), f, indent=2, ensure_ascii=False)

    totals = Counter(r["author"] for r in b.records)
    summary = {
        "totals": dict(totals),
        "by_dataset": {d: dict(Counter(r["author"] for r in b.records if r["dataset"] == d))
                       for d in sorted({r["dataset"] for r in b.records})},
        "by_generator": dict(Counter(r["generator_model"] for r in b.records if r["author"] == "llm").most_common()),
        "n_problems": len({r["problem_uid"] for r in b.records}),
        "n_problems_with_both": len({r["problem_uid"] for r in b.records if r["author"] == "llm"}
                                    & {r["problem_uid"] for r in b.records if r["author"] == "human"}),
        "cleaning_steps": dict(Counter((r["author"], s) for r in b.records for s in r["cleaning"]).most_common()),
        "steps_per_source": {k: dict(v) for k, v in b.log.items()},
        "n_borderline_dedup_pairs": len(b.registry.review),
        "category_source_by_problem": dict(Counter(
            {r["problem_uid"]: r["category_source"] for r in b.records}.values())),
        "category_by_problem": dict(Counter(
            {r["problem_uid"]: r["category"] for r in b.records}.values())),
    }
    summary["cleaning_steps"] = {f"{a}:{s}": n for (a, s), n in summary["cleaning_steps"].items()}
    with open(OUT_DIR / "full_data_build_log.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
