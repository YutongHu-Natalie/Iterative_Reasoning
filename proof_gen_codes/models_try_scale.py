import argparse
import json
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import time

# Qwen3 recommended params for thinking and non-thinking
THINKING_GEN_KWARGS = dict(temperature=0.6, top_p=0.95, top_k=20, min_p=0)
NON_THINKING_GEN_KWARGS = dict(temperature=0.7, top_p=0.8, top_k=20, min_p=0)
THINK_END_TOKEN_ID = 151668  # </think>

# Collect informal questions into a list
def load_qas():
    ds = load_from_disk("./Data/Choose_Models")
    qas = ds.select_columns(["id", "informal_theorem_qa"]).to_list()
    return qas
    



# Load the tokenizer and the model

model_path = "../models/Qwen3-8B"
def load_tokenizer_and_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto"
    )
    return tokenizer, model




# Build Prompts
def build_prompt(q):
    prompt = f"""
You are a mathematician tasked with proving or disproving mathematical statements with rigorous logic. You will be given a mathematical problem that requires a proof or disproof.

Here is the problem:
<problem>
{q}
</problem>

Your task is to determine whether the statement is true (and prove it) or false (and disprove it, typically with a counterexample or by proving the negation). Follow these important principles:

- Use rigorous mathematical logic and reasoning at every step
- Be systematic and thorough - do not skip steps or make unjustified leaps
- Identify relevant definitions, theorems, and properties that may be useful
- Do not hallucinate facts, theorems, or properties that you are not certain about
- If you are uncertain about a mathematical fact, acknowledge this limitation
- Verify your reasoning carefully, especially checking for logical gaps or errors
- For proofs, ensure each step follows logically from previous steps or known facts
- For disproof, a single valid counterexample is sufficient


After working through the problem in your scratchpad, provide your final answer with:
- A clear statement of whether you have proved or disproved the claim 
- A complete, step-by-step proof or disproof with clear reasoning at each step
- Concise but sufficient justification for each step

Format your response as follows:
<answer>

[Provide your complete proof or disproof with step-by-step reasoning]
</answer>

Remember: Mathematical rigor and correctness are paramount. It is better to acknowledge uncertainty than to present flawed reasoning.


"""
    return prompt

def generate_answer(question, tokenizer, model, enable_thinking=False):

    prompt = build_prompt(question)

    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(model.device)

    gen_kwargs = THINKING_GEN_KWARGS if enable_thinking else NON_THINKING_GEN_KWARGS

    outputs = model.generate(
        **inputs,
        max_new_tokens=2048,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        **gen_kwargs,
    )

    output_ids = outputs[0][inputs["input_ids"].shape[1]:].tolist()

    if enable_thinking:
        try:
            think_end = len(output_ids) - output_ids[::-1].index(THINK_END_TOKEN_ID)
        except ValueError:
            think_end = 0
        thinking = tokenizer.decode(output_ids[:think_end], skip_special_tokens=True).strip("\n")
        answer = tokenizer.decode(output_ids[think_end:], skip_special_tokens=True).strip("\n")
        return thinking, answer.strip()

    answer = tokenizer.decode(output_ids, skip_special_tokens=True)
    return None, answer.strip()



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable Qwen3 thinking mode (uses Temperature=0.6, TopP=0.95, TopK=20, MinP=0).",
    )
    return parser.parse_args()


def main():

    args = parse_args()

    qas = load_qas()

    tokenizer, model = load_tokenizer_and_model(model_path)

    results = []

    for sample in tqdm(qas):

        _, answer = generate_answer(
            sample["informal_theorem_qa"],
            tokenizer,
            model,
            enable_thinking=args.thinking,
        )

        results.append(
            {
                "id": sample["id"],
                "question": sample["informal_theorem_qa"],
                "answer": answer,
            }
        )

    out_path = "qwen3_8b_answers_thinking.json" if args.thinking else "qwen3_8b_answers.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
    





