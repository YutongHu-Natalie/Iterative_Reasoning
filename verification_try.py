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
We are given a covering map $ p: E \to X $, where $ X $ is a connected topological space, and we are to prove or disprove that all fibers $ p^{-1}(x) $ for $ x \in X $ have the same cardinality.

---

### Step 1: Recall the Definition of a Covering Map

A map $ p: E \to X $ is a **covering map** if for every point $ x \in X $, there exists an open neighborhood $ U $ of $ x $ such that $ p^{-1}(U) $ is a disjoint union of open sets in $ E $, each of which is mapped homeomorphically onto $ U $ by $ p $.

This means that the preimage of any point $ x \in X $, i.e., the fiber $ p^{-1}(x) $, is a discrete set (since it is the image of a discrete union under a homeomorphism), and each point in $ p^{-1}(x) $ has a neighborhood in $ E $ that maps homeomorphically to a neighborhood of $ x $ in $ X $.

---

### Step 2: Use the Connectedness of $ X $

Since $ X $ is connected, we can use the fact that the cardinality of the fibers is constant across $ X $, under certain conditions. This is a well-known result in covering space theory.

---

### Step 3: Use the Path Lifting Property

A covering map has the **path lifting property**: given a path $ \gamma: [0,1] \to X $ and a point $ e_0 \in p^{-1}(\gamma(0)) $, there exists a unique lift $ \tilde{\gamma}: [0,1] \to E $ such that $ p(\tilde{\gamma}(t)) = \gamma(t) $ and $ \tilde{\gamma}(0) = e_0 $.

This property implies that the fibers are not only discrete, but also that the number of points in each fiber is consistent across the space.

---

### Step 4: Use the Local Homeomorphism Property

Since $ p $ is a covering map, it is a local homeomorphism. That is, for every point $ e \in E $, there is an open neighborhood $ V \subset E $ of $ e $ such that $ p(V) $ is open in $ X $, and $ p|_V: V \to p(V) $ is a homeomorphism.

In particular, for any $ x \in X $, the fiber $ p^{-1}(x) $ is a discrete set, and the map $ p $ restricts to a bijection from each component of $ p^{-1}(U) $ to $ U $, for any neighborhood $ U $ of $ x $.

---

### Step 5: Use the Connectedness of $ X $ to Conclude Equal Fiber Cardinality

Now consider the function $ f: X \to \mathbb{N} \cup \{ \infty \} $, defined by $ f(x) = |p^{-1}(x)| $. We want to show that $ f $ is constant.

Let $ x_0 \in X $, and suppose that $ |p^{-1}(x_0)| = n $. Since $ X $ is connected, we can construct a path from $ x_0 $ to any other point $ x \in X $, and use the path lifting property to show that the fiber $ p^{-1}(x) $ must also have $ n $ elements.

This is a standard result in covering space theory, and it can be found in textbooks such as:

- *Algebraic Topology* by Allen Hatcher (Chapter 1, Section 1)
- *Topology* by James Munkres (Chapter 54, Section 1)

In particular, the key idea is that the fiber cardinality is constant on each path component of $ X $, and since $ X $ is connected, the fiber cardinality is constant everywhere.

---

### Conclusion

We have shown that for a covering map $ p: E \to X $, where $ X $ is connected, the fibers $ p^{-1}(x) $ for $ x \in X $ all have the same cardinality.

$$
\boxed{\text{The statement is true. All fibers } p^{-1}(x) \text{ have the same cardinality.}}
$$
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
