"""Reference implementation of consilience scoring, independent of any provider or framework.

Use it as the oracle when porting: the functions take plain Python lists, and
`python reference.py` runs the known-answer tests that any port must reproduce.

    S = C_final - alpha * C_initial      (Kong et al., "Consilience for Verifier-Free Test-Time Scaling")
"""
import math

ALPHA, WINDOW, SKIP, TOP_K = 3.0, 0.2, 0.05, 5
DELIMITERS = ("</think>", "<|channel|>final")   # earliest match wins; extend per system
EPS = 1e-9  # guards floor() against float error, e.g. 0.2 * 20 -> 4.000000000000001


def token_confidence(top_logprobs, k=TOP_K):
    """c_t = -(1/K) * sum of the top-K log-probs at one step. High when the distribution is peaked.

    `top_logprobs` are the log-probs of the K most likely candidate tokens at this step,
    NOT just the sampled token. Non-finite values (-inf / NaN sentinels) are dropped.
    Returns None when nothing usable is left.
    """
    vals = sorted((v for v in top_logprobs if v is not None and math.isfinite(v)), reverse=True)[:k]
    return -sum(vals) / len(vals) if vals else None


def split_reasoning(tokens, delimiters):
    """Index of the first token of the earliest delimiter found in the generated text, or None.

    Delimiters can straddle token boundaries, so search the concatenated text and map the
    character offset back to a token index. Reasoning = tokens[:index].
    """
    text = "".join(tokens)
    hits = [text.find(d) for d in delimiters if d and d in text]
    if not hits:
        return None
    pos, acc = min(hits), 0
    for i, t in enumerate(tokens):
        if acc + len(t) > pos:
            return i
        acc += len(t)
    return len(tokens)


def windows(L, window=WINDOW, skip=SKIP):
    """(P, W): skipped prefix length and window length for a segment of L tokens.

    window < 1 is a fraction of L; window >= 1 is an absolute token count.
    W is clamped to (L - P) // 2 so the two windows never overlap, and to at least 1.
    """
    P = math.floor(skip * L + EPS)
    W = math.floor(window * L + EPS) if window < 1 else int(window)
    W = max(1, min(W, (L - P) // 2))
    return P, W


def consilience_score(conf, alpha=ALPHA, window=WINDOW, skip=SKIP):
    """S = mean(conf[L-W:]) - alpha * mean(conf[P:P+W]). None for an empty sequence."""
    L = len(conf)
    if L == 0:
        return None
    P, W = windows(L, window, skip)
    init, final = conf[P:P + W], conf[L - W:]
    return sum(final) / len(final) - alpha * sum(init) / len(init)


def score_candidate(tokens, top_logprobs_per_token, delimiters=DELIMITERS,
                    alpha=ALPHA, window=WINDOW, skip=SKIP, k=TOP_K, min_tokens=20):
    """Score one sampled completion. Returns (score, scope, n_tokens); score is None if unscorable.

    scope is "reasoning" when a delimiter was found far enough in, otherwise "full".
    """
    pairs = [(t, token_confidence(lp, k)) for t, lp in zip(tokens, top_logprobs_per_token)]
    pairs = [(t, c) for t, c in pairs if c is not None]
    toks, conf = [t for t, _ in pairs], [c for _, c in pairs]
    cut = split_reasoning(toks, delimiters)
    scope = "full"
    if cut is not None and cut >= min_tokens:
        conf, scope = conf[:cut], "reasoning"
    return consilience_score(conf, alpha, window, skip), scope, len(conf)


def select(candidates, **kw):
    """candidates: list of (tokens, top_logprobs_per_token). Returns (best_index or None, scores).

    Ties go to the earliest candidate. Unscorable candidates are skipped; if none can be
    scored the caller should fall back to its first sample and warn.
    """
    scores = [score_candidate(t, lp, **kw)[0] for t, lp in candidates]
    valid = [(s, i) for i, s in enumerate(scores) if s is not None]
    if not valid:
        return None, scores
    best = max(valid, key=lambda x: (x[0], -x[1]))[1]
    return best, scores


# ---------------------------------------------------------------- known-answer tests
def _close(a, b, tol=1e-9):
    return a is not None and abs(a - b) <= tol


def _tests():
    # 1. token confidence is the negative mean of the top-K log-probs
    assert _close(token_confidence([-0.1, -2.0, -3.0, -4.0, -5.0]), 2.82)
    assert _close(token_confidence([-0.1, -2.0, -3.0, -4.0, -5.0, -9.0]), 2.82)    # keeps only top 5
    assert _close(token_confidence([-0.5, float("-inf")]), 0.5)                      # drops -inf
    assert token_confidence([]) is None
    # a peaked step is MORE confident than a flat one (the opposite of sampled-token NLL)
    peaked = [math.log(p) for p in (0.96, 0.01, 0.01, 0.01, 0.01)]
    flat = [math.log(0.2)] * 5
    assert token_confidence(peaked) > token_confidence(flat)

    # 2. windows: L=20 -> P=1, W=4; absolute window clamps to avoid overlap
    assert windows(20) == (1, 4)
    assert windows(100, window=2048) == (5, 47)
    assert windows(1) == (0, 1)

    # 3. the score itself: conf = 0..19 -> C_init = mean(1..4) = 2.5, C_final = mean(16..19) = 17.5
    conf = [float(i) for i in range(20)]
    assert _close(consilience_score(conf), 17.5 - 3.0 * 2.5)
    assert _close(consilience_score(conf, alpha=0.0), 17.5)
    assert consilience_score([]) is None

    # 4. delimiter split across tokens
    assert split_reasoning(["Let", " me", " think", "</th", "ink>", "42"], ["</think>"]) == 3
    assert split_reasoning(["no", " delimiter"], ["</think>"]) is None

    # 5. selection: "premature" starts and stays confident; "consilient" starts low and ends high
    def cand(conf_curve, answer_len=5):
        toks = ["x"] * len(conf_curve) + ["</think>"] + ["a"] * answer_len
        lps = [[-c] * 5 for c in conf_curve] + [[-20.0] * 5] * (answer_len + 1)   # c_t == conf value
        return toks, lps
    premature = [12.0] * 80 + [12.6] * 20
    consilient = [6.0] * 40 + [9.0] * 40 + [12.2] * 20
    pool = [cand(premature), cand(consilient)]
    best, scores = select(pool)
    assert best == 1, scores                                    # consilience picks the explorer
    assert select(pool, alpha=0.0)[0] == 0                       # alpha = 0 (last window only) does not
    s, scope, n = score_candidate(*cand(consilient))
    assert scope == "reasoning" and n == 100                     # answer tokens excluded

    # 6. the ratio form is NOT consilience: it is alpha-invariant
    ratio = lambda c, a: (sum(c[-20:]) / 20) / (a * sum(c[5:25]) / 20)
    assert (ratio(premature, 1) > ratio(consilient, 1)) == (ratio(premature, 5) > ratio(consilient, 5))
    print("all consilience reference tests passed")


if __name__ == "__main__":
    _tests()
