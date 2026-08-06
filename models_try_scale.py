import json
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import time

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

def generate_answer(question, tokenizer, model):

    prompt = build_prompt(question)

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=2048,
        do_sample=False,
        temperature=0.7,
        top_p=0.8,
        pad_token_id=tokenizer.eos_token_id,
    )

    answer = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    return answer.strip()



def main():

    qas = load_qas()

    tokenizer, model = load_tokenizer_and_model(model_path)

    results = []

    for sample in tqdm(qas):

        answer = generate_answer(
            sample["informal_theorem_qa"],
            tokenizer,
            model
        )

        results.append(
            {
                "id": sample["id"],
                "question": sample["informal_theorem_qa"],
                "answer": answer,
            }
        )

    with open("qwen3_8b_answers.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
    





