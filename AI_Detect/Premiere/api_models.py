"""Closed-source models (OpenAI / Anthropic APIs) for the detection scripts.

Shared by extract_structure.py and run_detection_api.py. A model key fixes
the provider, model id and reasoning effort; add a row to API_MODELS (and
its prices to PRICES) to try another model or effort level.

Credentials come from the environment: OPENAI_API_KEY for OpenAI models,
ANTHROPIC_API_KEY (or an `ant auth login` profile) for Claude models.
Needs openai>=2 (Responses API) and a recent anthropic SDK (output_config),
as pinned in requirements.txt.
"""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from tqdm import tqdm

API_MODELS = {
    # There is no gpt-6.1-luna; gpt-6-luna is the newest Luna.
    "gpt_6_luna_xhigh": dict(provider="openai", model="gpt-6-luna", effort="xhigh"),
    "gpt_6_1_sol_low": dict(provider="openai", model="gpt-6.1-sol", effort="low"),
    "claude_haiku_5_5": dict(provider="anthropic", model="claude-haiku-5-5", effort="medium"),
}

# USD per 1M tokens: (input, cached input, output). Standard short-context
# rates (gpt <=272K input, Haiku <=100K); reasoning/thinking tokens bill as
# output. Check the provider pricing pages before large runs.
PRICES = {
    "gpt-6-luna": (0.10, 0.01, 0.50),
    "gpt-6.1-sol": (2.00, 0.10, 10.00),
    "claude-haiku-5-5": (0.10, 0.01, 0.50),
}

_clients = {}


def _client(provider):
    # SDK clients are thread-safe; one per provider. Long timeout: xhigh
    # reasoning on a long proof can take many minutes.
    if provider not in _clients:
        if provider == "openai":
            import openai
            _clients[provider] = openai.OpenAI(timeout=1800, max_retries=5)
        else:
            import anthropic
            _clients[provider] = anthropic.Anthropic(timeout=1800, max_retries=5)
    return _clients[provider]


def cost_usd(model, usage):
    p_in, p_cached, p_out = PRICES[model]
    uncached = usage["input"] - usage["cached_input"]
    return (uncached * p_in + usage["cached_input"] * p_cached + usage["output"] * p_out) / 1e6


def _transient_errors(provider):
    """Errors still failing after the SDK's own retries: record them and move
    on (the item is redone on the next run). Anything else, e.g. a 400 from a
    bad request or a 401, is a config problem and is raised."""
    if provider == "openai":
        import openai as sdk
    else:
        import anthropic as sdk
    # APITimeoutError subclasses APIConnectionError in both SDKs.
    return (sdk.APIConnectionError, sdk.RateLimitError, sdk.InternalServerError)


def _call_openai(cfg, prompt, schema, max_output_tokens):
    kwargs = dict(model=cfg["model"], input=prompt, max_output_tokens=max_output_tokens,
                  reasoning={"effort": cfg["effort"]})
    if schema:
        kwargs["text"] = {"format": {"type": "json_schema", "name": schema["name"],
                                     "schema": schema["schema"], "strict": True}}
    resp = _client("openai").responses.create(**kwargs)
    u = resp.usage
    usage = {"input": u.input_tokens, "cached_input": u.input_tokens_details.cached_tokens or 0,
             "output": u.output_tokens, "reasoning": u.output_tokens_details.reasoning_tokens or 0}
    status = "ok"
    if resp.status == "incomplete":
        reason = getattr(resp.incomplete_details, "reason", None)
        status = "max_tokens" if reason == "max_output_tokens" else f"incomplete:{reason}"
    return {"text": resp.output_text, "thinking": None, "status": status, "usage": usage}


def _call_anthropic(cfg, prompt, schema, max_output_tokens):
    output_config = {"effort": cfg["effort"]}
    if schema:
        output_config["format"] = {"type": "json_schema", "schema": schema["schema"]}
    # Streaming: large max_tokens values are refused for non-streaming calls.
    with _client("anthropic").messages.stream(
        model=cfg["model"],
        max_tokens=max_output_tokens,
        thinking={"type": "adaptive", "display": "summarized"},
        output_config=output_config,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        msg = stream.get_final_message()
    u = msg.usage
    cached = u.cache_read_input_tokens or 0
    usage = {"input": u.input_tokens + cached + (u.cache_creation_input_tokens or 0),
             "cached_input": cached, "output": u.output_tokens, "reasoning": None}
    status = {"end_turn": "ok", "max_tokens": "max_tokens", "refusal": "refusal"}.get(msg.stop_reason, msg.stop_reason)
    if status == "refusal" and msg.stop_details:
        status = f"refusal:{msg.stop_details.category}"
    text = "".join(b.text for b in msg.content if b.type == "text")
    thinking = "\n".join(b.thinking for b in msg.content if b.type == "thinking") or None
    return {"text": text, "thinking": thinking, "status": status, "usage": usage}


def call_model(key, prompt, schema=None, max_output_tokens=64000):
    """One request. Returns {text, thinking, status, usage, cost_usd, api_error};
    status is "ok", "max_tokens", "refusal[:category]" or "api_error"."""
    cfg = API_MODELS[key]
    call = _call_openai if cfg["provider"] == "openai" else _call_anthropic
    try:
        out = call(cfg, prompt, schema, max_output_tokens)
    except _transient_errors(cfg["provider"]) as e:
        return {"text": None, "thinking": None, "status": "api_error", "usage": None,
                "cost_usd": 0.0, "api_error": f"{type(e).__name__}: {e}"}
    return {**out, "cost_usd": cost_usd(cfg["model"], out["usage"]), "api_error": None}


def model_config(key):
    cfg = API_MODELS[key]
    return {**cfg, "prices_per_1m": PRICES[cfg["model"]]}


class Budget:
    """Spending cap shared by every model in one invocation."""

    def __init__(self, limit_usd=None):
        self.limit = limit_usd
        self.spent = 0.0

    def exhausted(self):
        return self.limit is not None and self.spent >= self.limit


def run_pool(items, work, emit, workers, budget, desc):
    """Run work(item) -> record concurrently, calling emit(record) from this
    thread as results arrive. Stops submitting once the budget is spent (calls
    already in flight still finish, so it can overshoot by up to `workers`
    calls). Returns the number of items left unsubmitted."""
    pending = iter(items)
    left = len(items)
    with ThreadPoolExecutor(max_workers=workers) as pool, tqdm(total=len(items), desc=desc) as bar:
        running = set()

        def fill():
            nonlocal left
            while len(running) < workers and not budget.exhausted():
                item = next(pending, None)
                if item is None:
                    return
                running.add(pool.submit(work, item))
                left -= 1

        fill()
        while running:
            finished, _ = wait(running, return_when=FIRST_COMPLETED)
            for fut in finished:
                running.discard(fut)
                record = fut.result()  # re-raises config errors
                budget.spent += record.get("cost_usd") or 0.0
                emit(record)
                bar.update()
                bar.set_postfix(spent=f"${budget.spent:.3f}")
            fill()
    return left


def add_api_args(parser):
    parser.add_argument("--models", nargs="+", required=True, choices=list(API_MODELS) + ["all"])
    parser.add_argument("--workers", type=int, default=4, help="Concurrent API requests per model.")
    parser.add_argument("--max-cost", type=float, default=None,
                        help="Stop sending requests once this many USD have been spent in this invocation "
                             "(estimated from token usage and PRICES).")
    parser.add_argument("--max-output-tokens", type=int, default=64000,
                        help="Per request, reasoning tokens included. Only tokens actually used are billed.")
    parser.add_argument("--no-schema", action="store_true",
                        help="Don't request structured (JSON-schema) output; parse free text instead.")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--no-resume", action="store_true", help="Ignore saved results and start over.")


def print_cost(key, records):
    costs = [r.get("cost_usd") or 0.0 for r in records]
    if not costs:
        return
    avg = sum(costs) / len(costs)
    print(f"[{key}] ${sum(costs):.4f} over {len(costs)} items "
          f"(avg ${avg:.5f}/item, ~${avg * 1000:.2f} per 1000 items)")
