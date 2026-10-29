---
name: consilify
description: Consilify an LLM system, meaning add consilience, a verifier-free best-of-n selection rule computed from token log-probs (S = C_final − α·C_initial), to any LLM system, single-turn or agentic, on any provider, SDK, framework or language. Use when asked to consilify something, apply consilience, verifier-free test-time scaling, or confidence-based best-of-n selection to a codebase, model call, or agent loop.
---

# Consilify: add consilience to any LLM system

Sample several completions, score each one by **how its confidence evolves**, and keep the best. Good reasoning explores before it converges: low confidence early (several routes still open), high confidence at the end (the reasoning converged). Selecting by average confidence instead favours traces that committed too early, and that fails on hard problems.

Source: Kong, Hui, Mao, Huan, *Consilience for Verifier-Free Test-Time Scaling* (NeurIPS 2026). `reference.py` in this directory is the oracle for everything below; `python reference.py` runs its known-answer tests.

This skill is a specification, not an adapter. Read the target system, then implement the components below in its own language, client and conventions.

## The method (exact)

**Input per candidate:** for every generated token *t*, the log-probs of the **top-K most likely tokens** at that step (K = 5), plus the token strings (to find the reasoning boundary).

1. **Token confidence**
   `c_t = −(1/K) · Σ_{j=1..K} log p_t(v_j)`, where `v_1..v_K` are the K most likely tokens at step t.
   High when the distribution is peaked, because the runner-up tokens have tiny probabilities.
   - This is **not** the sampled token's log-prob. `−log p(sampled)` is surprisal and means the *opposite* (high = uncertain). If the system can only get the sampled token's log-prob, consilience cannot be computed. Say so and stop.
   - Drop non-finite values (−inf, NaN, sentinel values for tokens outside the sampler's support). If fewer than K values exist at a step, average the ones that do. Skip steps with none.

2. **Reasoning isolation:** score only the tokens *before* the answer or action.
   - Find the boundary by searching the **concatenated token text** for a delimiter and mapping the character offset back to a token index, because delimiters often span several tokens.
   - **Default delimiters:** `["</think>", "<|channel|>final"]`. Search for all of them and split at the earliest match.
     - `</think>` covers R1-style and Qwen3-style thinking models.
     - `<|channel|>final` covers the GPT-OSS harmony format.
     - Both are special tokens that don't appear in ordinary text, so the default is safe for any model. When neither is found, scoring falls back to the full sequence.
   - Add a delimiter for the target system when its reasoning ends somewhere else:
     - for agents: the start of the action, such as an action code fence (e.g. ```` ```bash ````) or the first tool-call token;
     - for non-thinking models: a marker the prompt asks for, such as `Final Answer:`.
     Plain-text markers can also appear inside the reasoning, so add them only when the prompt or output format guarantees them.
   - If no delimiter is found, or the reasoning is shorter than about 20 tokens, score the full sequence. This is the paper's `-full` variant: valid, but a weaker signal. Record which scope was used.

3. **Windows**, over the L reasoning tokens:
   - `P = floor(skip · L)`: the skipped prefix. The first tokens look nearly the same across samples of the same prompt.
   - `W = floor(window · L)` if `window < 1`, else `W = window` (an absolute token count).
   - Clamp `W` to `[1, (L − P) // 2]` so the windows never overlap. Add a tiny epsilon before `floor` so float error doesn't drop a token.
   - `C_initial = mean(c[P : P+W])`
   - `C_final = mean(c[L−W : L])`

4. **Score**
   `S = C_final − α · C_initial`
   It is a **difference**. `C_final / (α · C_initial)` is a known wrong variant: it ignores α and does not reproduce the paper.

5. **Select** `argmax S`. Ties go to the earliest sample.
   - Exclude candidates that can't be scored.
   - If none can be scored, return the first sample unchanged and warn. The likely cause is that the provider isn't returning top-K log-probs.

### Defaults

| Parameter | Default | Safe range | Note |
|---|---|---|---|
| α | 3.0 | 2–5 | 0 reduces to last-window confidence; very large α picks traces that never converge |
| window | 0.2 | 0.2–0.3, or 2048 tokens | fraction if < 1, token count if ≥ 1 |
| skip | 0.05 | | fraction of L |
| K (top log-probs) | 5 | | top-5 held 93.9% of the probability mass per step on GPT-OSS-20B / LiveCodeBench |
| n (samples) | 8 | | the paper uses 64 for single-turn and 32 at agent edit steps; gains already show at n = 8 |

## Configuration convention

Hard-code the defaults and let environment variables override them. Read the variables when a call is made, so they can change without a code edit. Parse them strictly: reject malformed values loudly rather than silently using a default.

| Variable | Default | Meaning |
|---|---|---|
| `CONSILIENCE_ENABLED` | `1` | `0` restores the original behaviour exactly |
| `CONSILIENCE_N` | `8` | samples per wrapped call |
| `CONSILIENCE_ALPHA` | `3.0` | α |
| `CONSILIENCE_WINDOW` | `0.2` | fraction (< 1) or token count (≥ 1) |
| `CONSILIENCE_SKIP` | `0.05` | skipped prefix fraction |
| `CONSILIENCE_TOP_K` | `5` | top log-probs requested and averaged |
| `CONSILIENCE_SCOPE` | `reasoning` | `reasoning` (isolate, falling back to full) or `full` |
| `CONSILIENCE_DELIMITERS` | `["</think>", "<|channel|>final"]` | JSON list of strings; replaces the default list (include the defaults if you still want them) |
| `CONSILIENCE_TRIGGER` | unset | agents only: regex that marks a step worth scaling (see below) |
| `CONSILIENCE_TRIGGER_MIN_LINES` | `0` | agents only: minimum action length, in lines, to trigger |
| `CONSILIENCE_LOG` | unset | path for one JSON line per wrapped call |

## When to ask the user

Use the defaults above for anything with a sensible default, and list what you assumed in the final report. Stop and ask only when the answer changes what you build and you can't settle it from the code, the docs or a test call.

| Situation | Ask about |
|---|---|
| The endpoint returns no top-K log-probs | Switch to another model or endpoint that does, or stop. |
| Log-probs cover only the visible answer, e.g. hidden reasoning | Proceed with the weaker `full` scope, or switch models. |
| Several plausible chokepoints, or unclear which call produces the final answer or action | Which call or calls to wrap. Name the candidates you found. |
| Reasoning boundary unclear: custom output format, no known delimiter | What marks the end of reasoning, or whether to add a prompted marker. |
| Agent with no obvious "critical step" in its action space | What should trigger scaling: which tools or actions, minimum size. |
| Cost: n × the wrapped calls per task is large, or there's no trigger | Acceptable n, and whether to scale every step or only triggered ones. |
| The caller uses temperature 0 or greedy decoding | Whether you may change the sampling settings for wrapped calls. |
| The caller streams | Pass streaming through unscaled, or buffer, select and replay. |
| The project has its own config system that conflicts with environment variables | Where the settings should live. |

Batch every open question into **one** message, before writing code. Make each question self-contained: say what you found, give the options, and lead with your recommendation. If the user is unavailable or says "use your judgement", take the recommended option and state it in the report.

## Integration procedure

### 1. Check feasibility first
Find out, from the code, the docs or one test call, whether the model and endpoint actually used:
- return **per-token top-K log-probs** (not just the chosen token's);
- return them for the **reasoning tokens**. Hidden-reasoning models only expose the visible answer, which leaves only the weak `full` scope;
- allow **sampling** (temperature > 0). Identical samples make selection meaningless.

If top-K log-probs are unavailable, or cover only the visible answer, ask the user how to proceed (see *When to ask the user* above). Never fake it with the sampled-token log-prob.

### 2. Find the chokepoint
Locate the call or calls that produce the **final answer or the agent's next action**. Prefer the single lowest-level function every such generation passes through, such as the model client method or the agent's `query`/`step`.

Leave auxiliary calls alone: embeddings, judges or graders, summarisers, title generation, parsers, retries of malformed output. If it's unclear which call produces the answer or action, ask the user.

### 3. Wrap it, preserving the interface
Give the wrapper the same signature and the same return type as the original, so nothing downstream changes. It should:
- add the provider's log-prob request with top-K ≥ `CONSILIENCE_TOP_K`, keeping every other caller parameter;
- obtain n samples. Use the provider's native multi-sample option where it exists (it usually shares the prompt cost); otherwise make n concurrent calls. Tolerate partial failures and select among the samples that succeeded;
- score each sample, select the best, and return it **in the original response shape**:
  - one candidate, at the position or index the caller expects;
  - log-prob payloads stripped if the caller didn't ask for them;
  - token usage and cost **summed over all samples**, so cost accounting stays honest;
- record a metadata line to `CONSILIENCE_LOG` when it is set: n requested and n scored, per-sample scores and scopes, selected index, α, window.

Pass the call through unchanged when:
- `CONSILIENCE_ENABLED=0`;
- the caller already requests multiple samples;
- the call streams. Either pass streaming through, or buffer the samples, select, and replay the winner as a stream; ask the user which one.

When the temperature is 0, pass through and warn that consilience needs sampling. Change the sampling settings only if the user agrees.

### 4. Agentic systems: scale only critical steps
Scaling every step multiplies the cost by n for little gain, because most steps just gather information. The paper's harness, which applies to any agent:
1. Sample **one** completion for the step as usual.
2. **Trigger check.** Is this a critical step? In the paper these are file-editing commands, matched by `sed -i|cat <<|tee |> /|patch |EOF`, with an action of at least L lines (L = 40–100). If not critical, execute it unchanged.
3. Sample **n − 1 more** for the same step, from the same history. The pool includes the first sample.
4. Keep candidates whose action is the **same kind** as the first: same matched keywords, same tool name. If none remain, keep all.
5. Select by consilience over each candidate's reasoning, meaning the text before its action, and continue the trajectory with the winner only. Discarded samples never enter the history.

Translate "critical step" to the agent's own action space: edit or write tools, final-answer or submit actions, long code blocks, irreversible operations. If the action space doesn't make this obvious, ask the user what should trigger scaling. Expose the rule through `CONSILIENCE_TRIGGER` / `CONSILIENCE_TRIGGER_MIN_LINES` or the system's own config mechanism. On SWE-bench Verified (n = 32) this raised GPT-OSS-120B from 23.0% to 26.9% and Qwen3-Coder-Next from 65.3% to 67.3%; the Qwen3 run took 18 h versus 8 h for the plain agent, not 32×.

### 5. Verify
1. **The scorer matches the spec.** Port the known-answer tests in `reference.py` (`_tests()`) to the target language, or call `reference.py` directly in Python, and make them pass. In particular: confidence is higher for a peaked step than a flat one; the windows match on L = 20; a low-to-high trace beats a flat-high trace at α = 3 but not at α = 0.
2. **Smoke run** with `CONSILIENCE_LOG` set:
   - n samples arrive and get valid scores;
   - scope is `reasoning`. If it is `full`, the delimiter is wrong or log-probs don't cover the reasoning;
   - samples differ;
   - the selected index is not always 0.
3. **The kill switch works:** `CONSILIENCE_ENABLED=0` gives byte-for-byte the original request and response handling.
4. Existing tests still pass.

### 6. Report to the user
Tell the user:
- which call was wrapped, and why that one;
- the delimiter and scope in use;
- the trigger rule, for agents;
- the environment variables;
- the expected cost: about n× generation tokens on wrapped calls, and how many calls are wrapped per task;
- any limitation found, such as log-probs only on visible tokens, a K cap, or no multi-sample support.

## Pitfalls
- Sampled-token log-prob used as "confidence". This inverts the signal.
- Answer tokens left in the scored segment. Restated conclusions are artificially confident and mute the signal.
- Inconsistent log-probs across candidates, e.g. some raw and some post-temperature. Request them the same way for every sample, and prefer raw model log-probs if the server lets you choose.
- Ratio instead of difference for S.
- Wrapping auxiliary calls, or every agent step, which wastes n× compute.
- Changing the caller's sampling parameters silently. If the temperature must change, say so.
