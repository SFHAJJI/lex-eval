"""lex-eval: run the cases against the live assistant, score them, log the run to MLflow, and optionally register the
evaluated configuration. Exits non-zero when a hard gate fails (a quotation that does not verify, a sentence outside the
templates, an error), so it can gate a release in CI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import mlflow

from .cases import dataset_sha256, load_cases
from .client import DEFAULT_BASE_URL, LexClient
from .score import CaseResult, aggregate, quotes_of, score_case, trace_of

MODEL_NAME = "lex-assistant"


def evaluate(client: LexClient, cases_path: Path, *, replay: bool) -> tuple[list[CaseResult], dict[str, Any]]:
    cases = load_cases(cases_path)
    coverage = client.coverage().body
    context = coverage.get("context") or {}
    results: list[CaseResult] = []
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
        if not first_trace and trace_of(first.body):
            first_trace = trace_of(first.body)
        print(f"{case.id:34} {result.expected:>24} -> {result.verdict!s:24} {'ok' if result.verdict_ok else 'MISS'}"
              f"  quotes {result.quotes_verified}/{result.quotes}  {result.seconds:.1f}s", flush=True)
    commits = context.get("commits") or {}
    freshness = {c.get("publisher"): c.get("newest_completed_observation")
                 for c in ((context.get("freshness") or {}).get("channels") or [])}
    snapshot = first_trace.get("snapshot") or {}
    facts = {
        "serving_commit": commits.get("serving_runtime_source_commit"),
        "index_builder_commit": commits.get("index_builder_source_commit"),
        "planner_id": first_trace.get("planner_id"),
        "templates_sha256": first_trace.get("templates_sha256"),
        "registry_sha256": first_trace.get("registry_sha256"),
        "snapshot_id": snapshot.get("snapshot_id"),
        "data_lu_legilux": freshness.get("lu-legilux"),
        "data_eu_eurlex": freshness.get("eu-eurlex"),
    }
    return results, facts


def gates(metrics: dict[str, float], min_accuracy: float) -> list[str]:
    failed = []
    if metrics["quote_verified_rate"] < 1.0:
        failed.append(f"quote_verified_rate {metrics['quote_verified_rate']:.3f} < 1")
    if metrics["non_template_strings"] > 0:
        failed.append(f"non_template_strings {metrics['non_template_strings']:.0f} > 0")
    if metrics["errors"] > 0:
        failed.append(f"errors {metrics['errors']:.0f} > 0")
    if metrics["verdict_accuracy"] < min_accuracy:
        failed.append(f"verdict_accuracy {metrics['verdict_accuracy']:.3f} < {min_accuracy}")
    return failed


def summary_markdown(results: list[CaseResult], metrics: dict[str, float], facts: dict[str, Any], failed: list[str]) -> str:
    lines = ["# lex-eval run", "", "## Release evaluated", ""]
    lines += [f"- {key}: {value}" for key, value in facts.items()]
    lines += ["", "## Metrics", ""] + [f"- {key}: {value:.3f}" if isinstance(value, float) else f"- {key}: {value}"
                                       for key, value in metrics.items()]
    lines += ["", "## Gates", "", "- all passed" if not failed else "", *[f"- FAILED: {item}" for item in failed]]
    lines += ["", "## Cases", "", "| case | type | expected | verdict | citation | quotes verified | same verdict | same plan |",
              "|---|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r.id} | {r.question_type} | {r.expected} | {r.verdict} | "
                     f"{'-' if r.citation_ok is None else r.citation_ok} | {r.quotes_verified}/{r.quotes} | {r.replay_identical} | "
                     f"{'-' if r.replay_same_plan is None else r.replay_same_plan} |")
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
    args = parser.parse_args(argv)

    with LexClient(args.base_url, min_interval=args.min_interval) as client:
        results, facts = evaluate(client, args.cases, replay=not args.no_replay)
    metrics = aggregate(results)
    failed = gates(metrics, args.min_accuracy)

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    run_name = f"{(facts.get('serving_commit') or 'unknown')[:8]}-{dataset_sha256(args.cases)[:8]}"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({"base_url": args.base_url, "dataset_sha256": dataset_sha256(args.cases),
                           "cases": len(results), "replay": not args.no_replay,
                           **{key: value for key, value in facts.items() if value is not None}})
        mlflow.log_metrics({key.replace(".", "_"): float(value) for key, value in metrics.items()})
        mlflow.set_tags({"gates": "passed" if not failed else "failed", "harness": "lex-eval 0.1.0"})
        summary = summary_markdown(results, metrics, facts, failed)
        mlflow.log_text(summary, "summary.md")
        mlflow.log_text("\n".join(json.dumps(r.row(), ensure_ascii=False) for r in results) + "\n", "results.jsonl")
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
