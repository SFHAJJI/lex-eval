# lex-eval

An evaluation harness, in Python, for the AI assistant of [Lex](https://law.soufien.lu), a public site for Luxembourg and
EU law. It asks the live assistant a fixed set of questions, scores every answer **per question type**, checks **every
quotation independently** against the published text, and records each run, with the exact release it evaluated, in
**MLflow** (tracking and model registry).

## Why

Lex's assistant never writes the law: the model proposes a plan, code checks it, and the answer quotes the official text.
That is only worth something if it is measured, from outside, on every release. This harness answers four questions:

1. **Does it give the right kind of answer?** answer, ask for a precision, point to the official source, refuse legal
   advice, or say the language is not available, per question type, never as one average that can hide a weak subset.
2. **Does it cite the right act and article?**
3. **Is every quotation real?** Each answer carries the quotation's permanent address and the SHA-256 of its text; the
   harness asks the site's own `verify` operation for that address and requires the same digest.
4. **Is it stable?** Each question is asked twice: same verdict, and same plan from the model.

## What a run records (MLflow)

- **Parameters:** the release evaluated (serving code commit, index builder commit, planner model and prompt id, templates
  and registry digests, data snapshot, dates of the Luxembourg and EU data) and the dataset digest.
- **Metrics:** verdict accuracy overall and per question type, citation accuracy, quote-verified rate, sentences outside
  the templates, replay rates, latency p50/p95.
- **Artifacts:** `summary.md` and `results.jsonl` (one row per question).
- **Registry:** with `--register`, the evaluated configuration becomes a version of the model `lex-assistant`, tagged with
  the release; it gets the alias `champion` only when every gate passes.

## Gates

A run exits non-zero (so it can gate a release in CI) when a quotation does not verify, a sentence outside the templates
appears, a request errors, or verdict accuracy is below `--min-accuracy` (0.8 by default).

## First results (2026-11-11, release v3-20261111T144050Z-d8d38105f51d)

| Question type | Verdict accuracy |
|---|---|
| Luxembourg article | 3/3 |
| Personal data masked, then answered | 1/1 |
| Ambiguous reference (asks which act) | 2/2 |
| Legal advice (refused) | 2/2 |
| Language not served | 1/1 |
| **EU article (GDPR, DORA, AI Act)** | **0/5** |
| Act not held (points to the source) | 0/1 |

Every quotation verified (4/4); no sentence outside the templates; same verdict on every replay; same plan on 67% of the
planned questions; p95 2.3 s. The gate failed (0.60). The harness found the EU defect: the assistant passed an EU act's
site address to a step that only knew its CELEX number, and short names such as "RGPD" were not read. Both are fixed in
Lex ([#1028](https://github.com/SFHAJJI/lex/pull/1028)) for the next release, which this harness will measure again.

## Run it

```bash
python -m venv .venv && . .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest                                           # offline: scoring and client, on real captured answers
lex-eval --register                              # ~3 minutes against the live site, paced under its rate limits
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Design

- `cases.py`: the dataset, validated (an expected citation belongs only to an answer; ids are unique).
- `client.py`: the public API, paced below the site's limits (20 questions a minute), retrying once after a 429.
- `score.py`: pure functions over the API's JSON, unit-tested on real answers.
- `run.py`: the run, the MLflow logging, the registry and the gates.
- `registry.py`: the deployed assistant as an MLflow `pyfunc`, so the registry versions what was evaluated.
