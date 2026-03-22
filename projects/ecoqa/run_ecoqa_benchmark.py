import argparse
import asyncio
import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .eval_runtime import build_engine, load_split, task_id


@dataclass
class ModelProfile:
    name: str
    kind: str
    source: str
    model: str
    base_url: str
    api_key: str
    tokenizer_model: str
    enabled: bool = True
    n_parallel_agents: int | None = None
    max_steps: int | None = None
    notes: str = ""


def _trajectory_is_correct(trajectory) -> bool:
    steps = getattr(trajectory, "steps", None) or []
    if steps:
        info = getattr(steps[-1], "info", {}) or {}
        if "is_correct" in info:
            return bool(info.get("is_correct"))
        metadata = info.get("metadata", {})
        if isinstance(metadata, dict):
            f1_score = metadata.get("f1_score")
            if f1_score is not None:
                try:
                    return float(f1_score) >= 1.0
                except (TypeError, ValueError):
                    pass
    return False


def _load_model_profiles(config_path: Path, include_disabled: bool = False) -> list[ModelProfile]:
    with open(config_path, encoding="utf-8") as f:
        payload = json.load(f)

    profiles: list[ModelProfile] = []
    for row in payload.get("models", []):
        enabled = bool(row.get("enabled", True))
        if not enabled and not include_disabled:
            continue

        api_key = str(row.get("api_key", "")).strip()
        api_key_env = str(row.get("api_key_env", "")).strip()
        if api_key_env:
            api_key = os.getenv(api_key_env, "").strip()

        profile = ModelProfile(
            name=str(row["name"]),
            kind=str(row.get("kind", "baseline")),
            source=str(row.get("source", "openai_compatible")),
            model=str(row["model"]),
            base_url=str(row["base_url"]),
            api_key=api_key or "None",
            tokenizer_model=str(row.get("tokenizer_model") or row["model"]),
            enabled=enabled,
            n_parallel_agents=row.get("n_parallel_agents"),
            max_steps=row.get("max_steps"),
            notes=str(row.get("notes", "")),
        )
        profiles.append(profile)
    return profiles


def _extract_metadata(trajectory) -> dict[str, Any]:
    if not getattr(trajectory, "steps", None):
        return {}
    info = getattr(trajectory.steps[-1], "info", {}) or {}
    metadata = info.get("metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def _extract_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _compute_metrics(trajectories: list) -> dict[str, Any]:
    if len(trajectories) == 0:
        return {
            "num_trajectories": 0,
            "num_unique_questions": 0,
            "pass_at_1": 0.0,
            "pass_at_k": 0.0,
            "f1_score_mean": 0.0,
            "exp_table_hit_rate_mean": 0.0,
            "exp_table_sql_succ_rate_mean": 0.0,
        }

    rows: list[dict[str, Any]] = []
    for traj in trajectories:
        reward = float(getattr(traj, "reward", 0.0) or 0.0)
        is_correct = _trajectory_is_correct(traj)
        task = getattr(traj, "task", {}) or {}
        metadata = _extract_metadata(traj)
        question_id = task_id(task)
        rows.append(
            {
                "question_id": question_id,
                "f1_score": _extract_float(metadata.get("f1_score"), reward),
                "exp_table_hit_rate": _extract_float(metadata.get("exp_table_hit_rate"), 0.0),
                "exp_table_sql_succ_rate": _extract_float(metadata.get("exp_table_sql_succ_rate"), 0.0),
                "is_correct": is_correct,
            }
        )

    pass_at_1 = sum(1 for r in rows if r["is_correct"]) / len(rows)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        grouped.setdefault(r["question_id"], []).append(r)
    pass_at_k = sum(1 for _, group in grouped.items() if any(g["is_correct"] for g in group)) / len(grouped)
    f1_score_mean = sum(r["f1_score"] for r in rows) / len(rows)
    exp_table_hit_rate_mean = sum(r["exp_table_hit_rate"] for r in rows) / len(rows)
    exp_table_sql_succ_rate_mean = sum(r["exp_table_sql_succ_rate"] for r in rows) / len(rows)

    return {
        "num_trajectories": len(rows),
        "num_unique_questions": len(grouped),
        "pass_at_1": pass_at_1,
        "pass_at_k": pass_at_k,
        "f1_score_mean": f1_score_mean,
        "exp_table_hit_rate_mean": exp_table_hit_rate_mean,
        "exp_table_sql_succ_rate_mean": exp_table_sql_succ_rate_mean,
    }


def _write_details(path: Path, run_id: str, profile: ModelProfile, trajectories: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for idx, traj in enumerate(trajectories):
            task = getattr(traj, "task", {}) or {}
            metadata = _extract_metadata(traj)
            reward = float(getattr(traj, "reward", 0.0) or 0.0)
            row = {
                "run_id": run_id,
                "model_name": profile.name,
                "model_kind": profile.kind,
                "trajectory_index": idx,
                "question_id": task_id(task),
                "f1_score": _extract_float(metadata.get("f1_score"), reward),
                "is_correct": _trajectory_is_correct(traj),
                "exp_table_hit_rate": _extract_float(metadata.get("exp_table_hit_rate"), 0.0),
                "exp_table_sql_succ_rate": _extract_float(metadata.get("exp_table_sql_succ_rate"), 0.0),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _append_summary(path: Path, summary_row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "timestamp_utc",
        "split",
        "repeat_n",
        "model_name",
        "model_kind",
        "model",
        "base_url",
        "num_trajectories",
        "num_unique_questions",
        "pass_at_1",
        "pass_at_k",
        "f1_score_mean",
        "exp_table_hit_rate_mean",
        "exp_table_sql_succ_rate_mean",
        "notes",
    ]
    if path.exists():
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            existing_header = reader.fieldnames or []
            existing_rows = list(reader)

        if existing_header != fieldnames:
            # Header schema changed. Rewrite
            # file with the latest schema and keep old rows when keys overlap.
            with open(path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in existing_rows:
                    writer.writerow({k: row.get(k, "") for k in fieldnames})
                writer.writerow({k: summary_row.get(k, "") for k in fieldnames})
            return

    exists = path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({k: summary_row.get(k, "") for k in fieldnames})


async def _run_single_profile(profile: ModelProfile, dataset, args) -> tuple[dict[str, Any], list]:
    engine = build_engine(
        model=profile.model,
        base_url=profile.base_url,
        api_key=profile.api_key,
        tokenizer_model=profile.tokenizer_model,
        temperature=args.temperature,
        top_p=args.top_p,
        n_parallel_agents=profile.n_parallel_agents or args.n_parallel_agents,
        max_steps=profile.max_steps or args.max_steps,
        max_prompt_length=args.max_prompt_length,
    )
    trajectories = await engine.execute_tasks(dataset)
    metrics = _compute_metrics(trajectories)
    return metrics, trajectories


def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare EcoQA models.")
    parser.add_argument("--models-config", default="projects/ecoqa/benchmarks/model_profiles.json")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--repeat-n", type=int, default=1, help="Repeat each task N times for pass@k.")
    parser.add_argument("--max-samples", type=int, default=0, help="Use first N samples for quick check. 0 means all.")
    parser.add_argument("--include-disabled", action="store_true", help="Run profiles marked enabled=false.")
    parser.add_argument("--output-dir", default="projects/ecoqa/benchmarks/results")
    parser.add_argument("--n-parallel-agents", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    args = parser.parse_args()

    config_path = Path(args.models_config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / "model_comparison.csv"

    profiles = _load_model_profiles(config_path, include_disabled=args.include_disabled)
    if not profiles:
        raise ValueError(f"No model profile is enabled in {config_path}.")

    dataset = load_split(args.split)
    if args.max_samples > 0:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    if args.repeat_n > 1:
        dataset = dataset.repeat(args.repeat_n)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    for profile in profiles:
        run_id = f"{timestamp}_{profile.name}"
        print(f"\n=== Running model: {profile.name} ({profile.model}) ===")
        metrics, trajectories = asyncio.run(_run_single_profile(profile, dataset, args))

        details_path = output_dir / f"{run_id}_details.jsonl"
        _write_details(details_path, run_id, profile, trajectories)

        summary_row = {
            "run_id": run_id,
            "timestamp_utc": timestamp,
            "split": args.split,
            "repeat_n": args.repeat_n,
            "model_name": profile.name,
            "model_kind": profile.kind,
            "model": profile.model,
            "base_url": profile.base_url,
            "num_trajectories": metrics["num_trajectories"],
            "num_unique_questions": metrics["num_unique_questions"],
            "pass_at_1": round(metrics["pass_at_1"], 6),
            "pass_at_k": round(metrics["pass_at_k"], 6),
            "f1_score_mean": round(metrics["f1_score_mean"], 6),
            "exp_table_hit_rate_mean": round(metrics["exp_table_hit_rate_mean"], 6),
            "exp_table_sql_succ_rate_mean": round(metrics["exp_table_sql_succ_rate_mean"], 6),
            "notes": profile.notes,
        }
        _append_summary(summary_csv, summary_row)

        print(
            f"pass@1={summary_row['pass_at_1']}, pass@k={summary_row['pass_at_k']}, "
            f"f1_mean={summary_row['f1_score_mean']}"
        )
        print(f"details -> {details_path}")

    print(f"\nSummary CSV updated: {summary_csv}")


if __name__ == "__main__":
    main()
