from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "../models/Qwen3-8B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)
problem = """
Let $p: E \to X$ be a covering map, where $X$ is a connected topological space. Prove that all fibers $p^{-1}(x)$ for $x \in X$ have the same cardinality.
"""
g = """
To prove this theorem, we can start by exploiting the local triviality inherent in the definition of a covering map.

First, let
\[
p \colon E \to X
\]
be a covering map, where \(X\) is a connected topological space. By the definition of a covering map, for every \(x \in X\) there exists an open neighborhood \(V\) of \(x\) and a discrete nonempty set \(I\) such that there is a homeomorphism
\[
\phi \colon p^{-1}(V) \to V \times I
\]
which commutes with the projection onto \(V\). This implies that for any point \(y \in V\), the fiber \(p^{-1}(y)\) is in bijection with \(I\) and hence
\[
\# p^{-1}(y) = \# I.
\]
In other words, the cardinality of the fiber is locally constant.

Now, define an equivalence relation \(\sim\) on \(X\) by
\[
x \sim x' \quad \Longleftrightarrow \quad \# p^{-1}(x) = \# p^{-1}(x').
\]
Clearly, this relation is reflexive, symmetric, and transitive. Moreover, the local triviality shows that if \(x \in X\) and \(V\) is an open neighborhood of \(x\) as described above, then for all \(y \in V\),
\[
\# p^{-1}(y) = \# p^{-1}(x),
\]
so every point \(y\) in \(V\) satisfies \(y \sim x\). This implies that each equivalence class of \(\sim\) is open in \(X\).

Since \(X\) is connected, it cannot be partitioned into two (or more) disjoint, nonempty open sets. Therefore, the only possibility is that there is exactly one equivalence class under \(\sim\). In other words, for any \(x, x' \in X\),
\[
\# p^{-1}(x) = \# p^{-1}(x'),
\]
so all fibers \(p^{-1}(x)\) have the same cardinality.

Thus, the theorem is proved.

"""
# prepare the model input
prompt = f"""
You are an expert mathematician grading a candidate proof for logical 
validity and completeness.

A proof can be sufficient with a wrong final answer (if the reasoning is complete and correct up to a localized error you should flag), and insufficient with a correct final answer (if it arrived there via invalid or unjustified steps).

A step counts as VALID only if it is justified explicitly enough for a 
careful reader to verify it without filling in unstated reasoning. An 
unstated but "obviously true" leap is a GAP, not a valid step — treat it 
the same as an outright error: mark the proof insufficient and localize it 
at that step.

QUESTION:
{problem}

CANDIDATE PROOF (steps are numbered):
{g}

TASK:
1. Work through the proof step by step. For each step, decide whether it 
   is explicitly justified by the problem statement and/or prior steps.
2. Write a short critique (2-5 sentences) identifying the first point of 
   failure, if any, and whether it is an ERROR (an invalid/false step) or 
   a TRUNCATION (the proof simply stops before reaching a justified 
   conclusion, with no invalid step along the way).
3. Output a structured judgment in exactly this format:

CRITIQUE: <your free-text critique>
SUFFICIENT: <yes|no>
EXTEND: <yes|no>
LOCALIZATION: <step number, or "N/A">

Disposition rules (apply exactly one):
- SUFFICIENT=yes → every step is explicitly justified and the proof is 
  complete. EXTEND=no, LOCALIZATION=N/A.
- SUFFICIENT=no, EXTEND=yes → every step so far is valid; the proof is 
  simply unfinished (TRUNCATION). The existing steps can stay as-is — the 
  next draft should pick up from where this one stopped and continue 
  forward. LOCALIZATION = the step after which reasoning stops.
- SUFFICIENT=no, EXTEND=no → the proof contains an ERROR or unjustified 
  GAP at some step. The existing steps from that point on cannot simply 
  be built on top of — the next draft should revise starting at the 
  localized step, discarding everything after it. LOCALIZATION = the step 
  where the error/gap occurs.



"""

messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
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
