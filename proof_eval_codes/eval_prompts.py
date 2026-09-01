"""LLM-judge prompts for scoring generated proofs in Results/*.json.

Two independent judgments are made per (question, model_answer) pair:

  - build_score_prompt: a weighted 0-1 rubric score (validity/completeness/
    correctness/clarity), for Results/eval_scores/.
  - build_pass_prompt: a step-by-step, binary valid/invalid verdict, for
    Results/eval_pass/.

Both ask the judge to return a single JSON object so responses can be parsed
without free-text scraping. Neither assumes the candidate ends its answer
with \\boxed{proved}/\\boxed{disproved} -- the generation prompts in
proof_gen_codes/ only ask models to "clearly state" whether they proved or
disproved the claim, so correctness is judged against that stated
conclusion (in whatever form it takes) versus the dataset's `truth_value`,
not against a specific boxed token.
"""

JSON_ONLY_INSTRUCTION = (
    r"Respond with a single JSON object and nothing else. Do not put the JSON "
    r"inside \boxed{}, do not use any LaTeX commands (\text{}, \left\{, "
    r"\right\}, etc.) anywhere in your response, and do not wrap it in "
    r"markdown code fences. Use plain JSON syntax throughout: double-quoted "
    r"keys and string values, and bare lowercase true/false for booleans."
)

SCORE_SYSTEM_PROMPT = (
    "You are an expert in scoring solutions for mathematical proof questions. "
    + JSON_ONLY_INSTRUCTION
)

PASS_SYSTEM_PROMPT = (
    "You are an expert in mathematical theorem proving and logical analysis. "
    + JSON_ONLY_INSTRUCTION
)


def ground_truth_text(truth_value: bool) -> str:
    if truth_value:
        return "TRUE -- the statement should be PROVED."
    return "FALSE -- the statement should be DISPROVED (e.g. via a counterexample or by proving the negation)."


def build_score_prompt(question: str, truth_value: bool, solution: str) -> str:
    ground_truth = ground_truth_text(truth_value)
    return f"""You are an expert in scoring solutions for mathematical proof questions. The following question asks to prove or disprove a statement, where the statement may be either true or false. The test subject was asked to prove the statement if it is true, or disprove it (e.g. with a counterexample or by proving the negation) if it is false, and to clearly state which of the two they concluded.

The question:
\"\"\"
{question}
\"\"\"

The ground truth of the statement:
\"\"\"
{ground_truth}
\"\"\"

The test subject's solution:
\"\"\"
{solution}
\"\"\"

Your task is to evaluate the proof's quality and assign a score from 0 to 1 based on four criteria: logical validity (40%), completeness (30%), correctness (20%), and clarity (10%).

Instructions:
1. Analyze the proof step by step.
2. For each criterion:
   - Logical Validity: Check if each step follows logically from the previous step, from a given assumption, or from a correctly cited definition/theorem. An unstated but "obviously true" leap is still a gap -- flag it.
   - Completeness: Verify that all necessary cases and steps are present to fully establish the conclusion, with no missing cases and no premature stopping.
   - Correctness: Confirm whether the test subject's final conclusion (proved vs. disproved, however it is stated) matches the ground truth above, and whether any final computation or claimed counterexample is actually correct.
   - Clarity: Assess whether the proof is clear, unambiguous, and well-explained.
3. Assign a sub-score (0 to 1) for each criterion and compute the total score using the weights:
   (0.4 x validity) + (0.3 x completeness) + (0.2 x correctness) + (0.1 x clarity).
4. Provide a brief explanation (2-3 sentences) summarizing any errors or issues and justifying the score.

Respond with exactly one JSON object and nothing else (no markdown code fences, no commentary outside the JSON, no \\boxed{{}}, no LaTeX commands anywhere in the response), in this format:
{{
  "score": <float>,
  "validity": <float>,
  "completeness": <float>,
  "correctness": <float>,
  "clarity": <float>,
  "explanation": "<string>"
}}
where "score" is the total weighted score, and "validity", "completeness", "correctness", "clarity" are the sub-scores, each between 0 and 1."""


def build_pass_prompt(question: str, truth_value: bool, solution: str) -> str:
    ground_truth = ground_truth_text(truth_value)
    return f"""You are an expert in mathematical theorem proving and logical analysis. Given the following theorem and a candidate proof or disproof of it, break the candidate's response into its constituent reasoning steps and determine, step by step, whether each one is valid.

# Theorem
\"\"\"
{question}
\"\"\"

# Ground truth
\"\"\"
{ground_truth}
\"\"\"

# Candidate proof or disproof
\"\"\"
{solution}
\"\"\"

# Instructions
1. Split the candidate's response into its individual reasoning steps. If the response does not already number its steps, segment it into logical steps yourself.
2. For each step:
   - Verify it is mathematically correct, logically sound, and relevant to proving or disproving the theorem.
   - Check for correct use of any definitions, theorems, or properties it cites.
   - Ensure it follows from previous steps or given assumptions without an unstated logical gap.
   - If the response claims a disproof, confirm it correctly demonstrates a counterexample or a contradiction.
3. Form an overall judgment:
   - The response is valid only if every step is valid AND its final conclusion (proved vs. disproved) matches the ground truth above.
   - If it is invalid, identify the critical error(s) and how to fix them.

Respond with exactly one JSON object and nothing else (no markdown code fences, no commentary outside the JSON, no \\boxed{{}}, no LaTeX commands anywhere in the response), in this format:
{{
  "steps": [
    {{"step": <int>, "description": "<short paraphrase of the step>", "valid": <true|false>, "justification": "<why this step is or is not valid>"}}
  ],
  "valid": <true|false>,
  "summary": "<if valid: 1-2 sentences confirming the proof fully addresses the theorem. if invalid: 1-3 sentences summarizing the critical error(s) and recommending how to fix them>"
}}"""
