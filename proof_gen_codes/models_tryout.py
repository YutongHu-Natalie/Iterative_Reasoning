from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "../../models/Qwen3-8B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)
problem = """
Let $H \subset \mathbb{C}^n$ be an open set and let $V \subset H$ be an analytic set of codimension $\geq 2$ in $H$. Suppose $f: H \setminus V \to \mathbb{C}$ is a holomorphic function. Prove that $f$ has a unique holomorphic extension to the whole set $H$."""

# prepare the model input
prompt = f"""
You are a mathematician tasked with proving or disproving mathematical statements with rigorous logic. You will be given a mathematical problem that requires a proof or disproof.

Here is the problem:
<problem>
{problem}
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
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

# parsing thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)
