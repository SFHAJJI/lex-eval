"""lex-eval: run the cases against the live assistant, score them, log the run to MLflow, and optionally register the
evaluated configuration. Exits non-zero when a hard gate fails (a quotation that does not verify, a sentence outside the
templates, an error), so it can gate a release in CI."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import mlflow

from .cases import dataset_sha256, load_cases
from .client import DEFAULT_BASE_URL, LexClient
from .score import OUTCOMES, CaseResult, aggregate, quotes_of, score_case, search_step, trace_of, trace_record

# The planner's model, at the price per million tokens the deployment states (deploy/main.bicep's assistant settings, in
# euros): an estimate, as the site does not report the tokens it used.
PRICE_IN_PER_MTOK_EUR = 0.40
PRICE_OUT_PER_MTOK_EUR = 1.60
# The planner prompt's size in characters (src/Lex.V3.Api/assistant-planner-prompt.md), and what the site adds around the
# question (the dates, references and languages it reads first), both in characters: four make about one token.
PROMPT_CHARS = 7770
INPUT_OVERHEAD_CHARS = 1200


def harness_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], capture_output=True,
                              text=True, check=True).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def cost_estimate(question: str, plan: Any) -> float:
    """An estimate of one planned question's cost in euros: the prompt, the question and what the site adds, in; the
    plan, out; four characters to a token, at the model's list price. Zero when no plan was made."""
    if not plan:
        return 0.0
    tokens_in = (PROMPT_CHARS + INPUT_OVERHEAD_CHARS + len(question)) / 4
    tokens_out = len(json.dumps(plan, ensure_ascii=False)) / 4
    return tokens_in * PRICE_IN_PER_MTOK_EUR / 1e6 + tokens_out * PRICE_OUT_PER_MTOK_EUR / 1e6

MODEL_NAME = "lex-assistant"


def evaluate(client: LexClient, cases_path: Path, *, replay: bool) -> tuple[list[CaseResult], dict[str, Any], list[dict[str, Any]]]:
    cases = load_cases(cases_path)
    coverage = client.coverage().body
    context = coverage.get("context") or {}
    results: list[CaseResult] = []
    traces: list[dict[str, Any]] = []
    first_trace: dict[str, Any] = {}
    for case in cases:
        first = client.ask(case.question)
        second = client.ask(case.question).body if replay else None
        verifications = {}
        for quote in quotes_of(first.body):
            if quote.permalink not in verifications:
                verifications[quote.permalink] = client.verify(quote.permalink).body
        result = score_case(case, first.body, first.seconds, replay=second, verifications=verifications, status=first.status)
        results.append(result)
        # The plan's search asked again, as the trace records it, for its first hits and its best-hit decision.
        asked = search_step(first.body)
        search = client.call("search", asked).body if asked is not None else None
        record = trace_record(case, result, first.body, search)
        record["cost_eur_estimate"] = cost_estimate(case.question, (trace_of(first.body) or {}).get("plan"))
        traces.append(record)
        if not first_trace and trace_of(first.body):
            first_trace = trace_of(first.body)
        print(f"{case.id:38} {result.expected:>10} -> {result.verdict!s:10} {result.outcome:16}"
              f" {result.chosen_article or '-':>10}  quotes {result.quotes_verified}/{result.quotes}  {result.seconds:.1f}s", flush=True)
    commits = context.get("commits") or {}
    freshness = {c.get("publisher"): c.get("newest_completed_observation")
                 for c in ((context.get("freshness") or {}).get("channels") or [])}
    snapshot = first_trace.get("snapshot") or {}
    planner = str(first_trace.get("planner_id") or "")
    planner_parts = planner.split("/")
    facts = {
        "harness_commit": harness_commit(),
        "serving_commit": commits.get("serving_runtime_source_commit"),
        "index_builder_commit": commits.get("index_builder_source_commit"),
        "snapshot_sha256": snapshot.get("snapshot_sha256"),
        "planner_id": first_trace.get("planner_id"),
        "model_deployment": planner_parts[1] if len(planner_parts) > 2 else None,
        "planner_prompt_version": planner_parts[2] if len(planner_parts) > 2 else None,
        "templates_sha256": first_trace.get("templates_sha256"),
        "registry_sha256": first_trace.get("registry_sha256"),
        "snapshot_id": snapshot.get("snapshot_id"),
        "data_lu_legilux": freshness.get("lu-legilux"),
        "data_eu_eurlex": freshness.get("eu-eurlex"),
    }
    return results, facts, traces


def regressions(results: list[CaseResult], baseline: Path | None) -> list[str]:
    """The cases a previous run passed (the right article, or the right behaviour) that this run does not."""
    if baseline is None:
        return []
    passed = {}
    for line in baseline.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            passed[row["id"]] = row.get("outcome") in {"right_article", "right_behaviour"} or (
                row.get("outcome") is None and row.get("verdict_ok") and row.get("citation_ok") in (None, True))
    return [r.id for r in results if passed.get(r.id) and r.outcome not in {"right_article", "right_behaviour"}]


def gates(metrics: dict[str, float], min_accuracy: float, *, gate_suite: str | None = None, regressed: list[str] | None = None) -> list[str]:
    failed = []
    # The headline: no answer quotes a wrong article on the gated suite (the 1997 insurance-contract law).
    if gate_suite is not None and metrics.get(f"outcome.wrong_article.{gate_suite}", 0) > 0:
        failed.append(f"wrong_article on {gate_suite} {metrics[f'outcome.wrong_article.{gate_suite}']:.0f} > 0")
    if regressed:
        failed.append(f"previously passing cases fail: {', '.join(regressed)}")
    if metrics["quote_verified_rate"] < 1.0:
        failed.append(f"quote_verified_rate {metrics['quote_verified_rate']:.3f} < 1")
    if metrics["non_template_strings"] > 0:
        failed.append(f"non_template_strings {metrics['non_template_strings']:.0f} > 0")
    if metrics["errors"] > 0:
        failed.append(f"errors {metrics['errors']:.0f} > 0")
    accuracy = metrics.get("verdict_accuracy_floor_set", metrics["verdict_accuracy"])
    if accuracy < min_accuracy:
        name = "verdict_accuracy_floor_set" if "verdict_accuracy_floor_set" in metrics else "verdict_accuracy"
        failed.append(f"{name} {accuracy:.3f} < {min_accuracy}")
    return failed


def summary_markdown(results: list[CaseResult], metrics: dict[str, float], facts: dict[str, Any], failed: list[str]) -> str:
    lines = ["# lex-eval run", "", "## Release evaluated", ""]
    lines += [f"- {key}: {value}" for key, value in facts.items()]
    lines += ["", "## Metrics", ""] + [f"- {key}: {value:.3f}" if isinstance(value, float) else f"- {key}: {value}"
                                       for key, value in metrics.items()]
    lines += ["", "## Gates", "", "- all passed" if not failed else "", *[f"- FAILED: {item}" for item in failed]]
    lines += ["", "## Results by kind", "", "| kind | cases |", "|---|---|"]
    lines += [f"| {outcome} | {sum(1 for r in results if r.outcome == outcome)} |" for outcome in OUTCOMES]
    lines += ["", "## Cases", "", "| case | suite | expected | verdict | kind | article chosen | quotes verified | same verdict | same plan |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r.id} | {r.suite} | {r.expected} | {r.verdict} | {r.outcome} | {r.chosen_article or '-'} | "
                     f"{r.quotes_verified}/{r.quotes} | {r.replay_identical} | {'-' if r.replay_same_plan is None else r.replay_same_plan} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lex-eval", description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--cases", type=Path, default=Path("cases/lex-assistant-v1.jsonl"))
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--experiment", default="lex-assistant")
    parser.add_argument("--no-replay", action="store_true", help="ask each question once (no determinism check)")
    parser.add_argument("--register", action="store_true", help="register the evaluated configuration in the model registry")
    parser.add_argument("--min-accuracy", type=float, default=0.8)
    parser.add_argument("--min-interval", type=float, default=3.5, help="seconds between requests (the site allows 20 asks a minute)")
    parser.add_argument("--gate-suite", default="insurance", help="the suite on which one wrong article fails the run ('' for none)")
    parser.add_argument("--baseline", type=Path, default=None, help="a previous run's results.jsonl: a case it passed must pass")
    parser.add_argument("--split", default="dev", help="what the cases are: dev (the set the work is checked on) or heldout")
    parser.add_argument("--out-dir", type=Path, default=None, help="also write summary.md, results.jsonl and traces.jsonl here")
    parser.add_argument("--accuracy-cases", type=Path, default=None,
                        help="apply --min-accuracy to the cases of this file only (the set the floor was set on), by id")
    args = parser.parse_args(argv)

    with LexClient(args.base_url, min_interval=args.min_interval) as client:
        results, facts, traces = evaluate(client, args.cases, replay=not args.no_replay)
    metrics = aggregate(results)
    planned = [t["cost_eur_estimate"] for t in traces if t["cost_eur_estimate"] > 0]
    metrics["cost_eur_per_question_estimate"] = sum(planned) / len(planned) if planned else 0.0
    regressed = regressions(results, args.baseline)
    if args.accuracy_cases is not None:
        # The verdict floor was set on its own set (v1's 16 questions): it is held there, word for word, and the larger set
        # is held by its own gates (no wrong article on the gated suite, no regression).
        floor_ids = {case.id for case in load_cases(args.accuracy_cases)}
        floor = [r for r in results if r.id in floor_ids]
        if len(floor) != len(floor_ids):
            raise SystemExit(f"--accuracy-cases names {len(floor_ids)} cases and this run holds {len(floor)} of them")
        metrics["verdict_accuracy_floor_set"] = sum(r.verdict_ok for r in floor) / len(floor)
    failed = gates(metrics, args.min_accuracy, gate_suite=args.gate_suite or None, regressed=regressed)

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    run_name = f"{(facts.get('serving_commit') or 'unknown')[:8]}-{dataset_sha256(args.cases)[:8]}"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({"base_url": args.base_url, "dataset_sha256": dataset_sha256(args.cases), "dataset": args.cases.name,
                           "cases": len(results), "replay": not args.no_replay, "split": args.split,
                           "gate_suite": args.gate_suite, "baseline": str(args.baseline) if args.baseline else "",
                           "price_in_per_mtok_eur": PRICE_IN_PER_MTOK_EUR, "price_out_per_mtok_eur": PRICE_OUT_PER_MTOK_EUR,
                           **{key: value for key, value in facts.items() if value is not None}})
        mlflow.log_metrics({key.replace(".", "_"): float(value) for key, value in metrics.items()})
        mlflow.set_tags({"gates": "passed" if not failed else "failed", "harness": "lex-eval 0.2.0", "split": args.split})
        # One trace per question: the plan, each operation, the search's first hits and best hit, the article chosen.
        mlflow.log_text("\n".join(json.dumps(t, ensure_ascii=False) for t in traces) + "\n", "traces.jsonl")
        summary = summary_markdown(results, metrics, facts, failed)
        mlflow.log_text(summary, "summary.md")
        mlflow.log_text("\n".join(json.dumps(r.row(), ensure_ascii=False) for r in results) + "\n", "results.jsonl")
        if args.out_dir is not None:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            (args.out_dir / "summary.md").write_text(summary, encoding="utf-8")
            (args.out_dir / "results.jsonl").write_text("\n".join(json.dumps(r.row(), ensure_ascii=False) for r in results) + "\n", encoding="utf-8")
            (args.out_dir / "traces.jsonl").write_text("\n".join(json.dumps(t, ensure_ascii=False) for t in traces) + "\n", encoding="utf-8")
            (args.out_dir / "run.json").write_text(json.dumps({"run_id": run.info.run_id, "metrics": metrics, "facts": facts,
                                                              "failed": failed}, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.register:
            from .registry import LexAssistantModel

            mlflow.pyfunc.log_model("model", python_model=LexAssistantModel(args.base_url))
            version = mlflow.register_model(f"runs:/{run.info.run_id}/model", MODEL_NAME)
            registry = mlflow.MlflowClient()
            for key in ("serving_commit", "planner_id", "snapshot_id"):
                if facts.get(key):
                    registry.set_model_version_tag(MODEL_NAME, version.version, key, str(facts[key]))
            registry.set_model_version_tag(MODEL_NAME, version.version, "gates", "passed" if not failed else "failed")
            if not failed:
                registry.set_registered_model_alias(MODEL_NAME, "champion", version.version)
            print(f"registered {MODEL_NAME} version {version.version}{' as champion' if not failed else ''}")
    print(summary)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
