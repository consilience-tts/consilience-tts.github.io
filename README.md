# consilience-tts.github.io

Project page for **Consilience for Verifier-Free Test-Time Scaling** (NeurIPS 2026), live at **https://consilience-tts.github.io**, and home of the **consilify** skill.

> Good answers explore before they converge. Consilience samples several reasoning traces and keeps the one that starts uncertain and ends certain, scored from token log-probs alone: `S = C_final − α · C_initial`. No verifier, no voting, no training.

## consilify: add consilience to any LLM system

`consilify` is a [Claude Code](https://docs.claude.com/en/docs/claude-code) skill. It gives the agent the exact method, the defaults and an integration procedure. The agent then adapts consilience to your system, single-turn or agentic, whatever the provider, SDK or language:

1. checks the model exposes per-token top-K log-probs;
2. finds the call that produces the answer or agent action;
3. wraps it without changing its interface;
4. verifies the scorer against known-answer tests.

It asks you when a decision is yours, such as which call to wrap or what counts as a critical agent step.

### Install

```bash
curl -LO https://consilience-tts.github.io/skill/consilify.zip
unzip consilify.zip -d ~/.claude/skills/
```

Then, in your project: `claude "consilify this agent"` (or ask for it in any Claude Code session).

### Configure

Defaults follow the paper. Override them with environment variables on the consilified system:

| Variable | Default | Meaning |
|---|---|---|
| `CONSILIENCE_ENABLED` | `1` | `0` restores the original behaviour |
| `CONSILIENCE_N` | `8` | samples per wrapped call |
| `CONSILIENCE_ALPHA` | `3.0` | penalty on early confidence (2–5 is safe) |
| `CONSILIENCE_WINDOW` | `0.2` | window as a fraction (< 1) or token count (≥ 1) |
| `CONSILIENCE_SKIP` | `0.05` | skipped prefix fraction |
| `CONSILIENCE_TOP_K` | `5` | top log-probs per token |
| `CONSILIENCE_DELIMITERS` | `["</think>", "<\|channel\|>final"]` | end-of-reasoning markers (JSON list) |
| `CONSILIENCE_TRIGGER`, `CONSILIENCE_TRIGGER_MIN_LINES` | unset, `0` | agents: which steps get scaled |
| `CONSILIENCE_LOG` | unset | JSON-lines log of per-call scores |

The full specification is in [`skill/consilify/SKILL.md`](skill/consilify/SKILL.md). [`skill/consilify/reference.py`](skill/consilify/reference.py) is a plain-Python reference scorer: `python3 skill/consilify/reference.py` runs its known-answer tests.

## Repository layout

```
index.html               project page (vanilla HTML/CSS/JS, no build step)
static/data.js           chart data re-extracted from the paper's figures
skill/consilify/         the skill: SKILL.md + reference.py
skill/consilify.zip      download served by the page
tools/                   scripts that regenerate static/data.js
```

## Updating

- **Page:** edit `index.html` and open it directly in a browser to preview.
- **Skill:** edit `skill/consilify/`, run `python3 skill/consilify/reference.py`, then rebuild the zip:
  ```bash
  cd skill && rm -f consilify.zip && zip -rq -X consilify.zip consilify -x '*/__pycache__/*' '*.DS_Store'
  ```
- **Chart data:** `python3 tools/extract_figures.py /path/to/paper/final_figure` (needs poppler's `pdftocairo`).

GitHub Pages serves the `main` branch root; `.nojekyll` keeps files served as-is.

## Citation

```bibtex
@inproceedings{kong2026consilience,
  title     = {Consilience for Verifier-Free Test-Time Scaling},
  author    = {Kong, Lecheng and Hui, Like and Mao, Haitao and Huan, Jun},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

MIT, see [LICENSE](LICENSE).
