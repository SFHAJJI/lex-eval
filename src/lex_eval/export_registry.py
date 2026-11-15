"""Export the MLflow model registry of the Lex assistant as JSON, for the site's "How it works" page (the owner's order of
2026-11-15: MLflow visible in the app). Read-only: it asks MLflow's own client for the registered model, its aliases and
every version with its tags, and for each version's run its parameters, metrics and logged artifacts; it writes them as
they are. Usage: python -m lex_eval.export_registry [--tracking-uri sqlite:///mlflow.db] OUT.json"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import mlflow
from mlflow import MlflowClient

from .run import MODEL_NAME

METRICS = (
    "cases", "verdict_accuracy", "citation_accuracy", "quote_verified_rate", "replay_same_verdict_rate", "replay_same_plan_rate",
    "latency_p50_s", "latency_p95_s", "outcome_right_article", "outcome_wrong_article", "outcome_not_found",
    "outcome_wrongly_refused", "outcome_right_behaviour", "cost_eur_per_question_estimate",
)


def _date(ms: int | None) -> str | None:
    return None if ms is None else datetime.fromtimestamp(ms / 1000, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def export(tracking_uri: str) -> dict:
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    model = client.get_registered_model(MODEL_NAME)
    aliases = dict(model.aliases or {})
    versions = []
    for version in sorted(client.search_model_versions(f"name='{MODEL_NAME}'"), key=lambda v: int(v.version)):
        run = client.get_run(version.run_id)
        artifacts = sorted(item.path for item in client.list_artifacts(version.run_id))
        versions.append({
            "version": int(version.version),
            "created": _date(version.creation_timestamp),
            "source": version.source,
            "run_id": version.run_id,
            "run_name": run.info.run_name,
            "tags": dict(version.tags or {}),
            "aliases": sorted(name for name, number in aliases.items() if str(number) == str(version.version)),
            "params": {key: run.data.params[key] for key in ("dataset", "dataset_sha256", "base_url") if key in run.data.params},
            "metrics": {key: run.data.metrics[key] for key in METRICS if key in run.data.metrics},
            "artifacts": artifacts,
        })
    experiment = client.get_experiment_by_name("lex-assistant")
    runs = client.search_runs([experiment.experiment_id], max_results=1000) if experiment else []
    return {
        "mlflow_version": mlflow.__version__,
        "model": MODEL_NAME,
        "aliases": {name: int(number) for name, number in aliases.items()},
        "versions": versions,
        "experiment": experiment.name if experiment else None,
        "runs_logged": len(runs),
        "exported": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    args = parser.parse_args()
    args.out.write_text(json.dumps(export(args.tracking_uri), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
