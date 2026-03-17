import asyncio
import json
import os
from pathlib import Path

from .eval_runtime import build_engine, load_split, task_id


def _trajectory_is_correct(trajectory) -> bool:
    steps = getattr(trajectory, "steps", None) or []
    if steps:
        info = getattr(steps[-1], "info", {}) or {}
        if "is_correct" in info:
            return bool(info.get("is_correct"))
        metadata = info.get("metadata", {})
        if isinstance(metadata, dict):
            correctness_reward = metadata.get("correctness_reward")
            if correctness_reward is not None:
                try:
                    return float(correctness_reward) >= 1.0
                except (TypeError, ValueError):
                    pass
    return False


def _extract_float_metadata(trajectory, key: str) -> float | None:
    steps = getattr(trajectory, "steps", None) or []
    if not steps:
        return None
    info = getattr(steps[-1], "info", {}) or {}
    metadata = info.get("metadata", {})
    if not isinstance(metadata, dict):
        return None
    value = metadata.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _print_eval_metrics(trajectories: list) -> None:
    if not trajectories:
        print("Total unique problems: 0")
        print("Pass@1: 0.0")
        print("Pass@k: 0.0")
        print("Expected Table Hit Rate: 0.0")
        print("Expected Table SQL Success Rate: 0.0")
        return

    grouped: dict[str, list[bool]] = {}
    total_correct = 0
    hit_rates: list[float] = []
    sql_succ_rates: list[float] = []
    for traj in trajectories:
        task = getattr(traj, "task", {}) or {}
        qid = task_id(task)
        is_correct = _trajectory_is_correct(traj)
        grouped.setdefault(qid, []).append(is_correct)
        if is_correct:
            total_correct += 1
        hit_rate = _extract_float_metadata(traj, "exp_table_hit_rate")
        if hit_rate is not None:
            hit_rates.append(hit_rate)
        sql_succ_rate = _extract_float_metadata(traj, "exp_table_sql_succ_rate")
        if sql_succ_rate is not None:
            sql_succ_rates.append(sql_succ_rate)

    pass_at_1 = total_correct / len(trajectories)
    pass_at_k = sum(1 for group in grouped.values() if any(group)) / len(grouped)
    avg_hit_rate = sum(hit_rates) / len(hit_rates) if hit_rates else 0.0
    avg_sql_succ_rate = sum(sql_succ_rates) / len(sql_succ_rates) if sql_succ_rates else 0.0

    print("Total unique problems:", len(grouped))
    print("Pass@1:", pass_at_1)
    print("Pass@k:", pass_at_k)
    print("Expected Table Hit Rate:", avg_hit_rate)
    print("Expected Table SQL Success Rate:", avg_sql_succ_rate)


def _infer_termination_reason(trajectory, max_steps: int) -> str:
    steps = getattr(trajectory, "steps", None) or []
    if not steps:
        return "UNKNOWN"

    last = steps[-1]
    if bool(getattr(last, "done", False)):
        return "ENV_DONE_OR_EARLY_STOP"
    if len(steps) >= max_steps:
        return "MAX_STEPS"
    return "UNKNOWN"


def _write_steps_jsonl(path: str, trajectories: list, max_steps: int) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for idx, traj in enumerate(trajectories):
            task = getattr(traj, "task", {}) or {}
            steps = getattr(traj, "steps", []) or []
            row = {
                "trajectory_index": idx,
                "question_id": task_id(task),
                "question": task.get("question", ""),
                "question_type": task.get("question_type", ""),
                "answer_type": task.get("answer_type", ""),
                "ground_truth": task.get("ground_truth", ""),
                "reward": float(getattr(traj, "reward", 0.0) or 0.0),
                "num_steps": len(steps),
                "termination_reason_inferred": _infer_termination_reason(traj, max_steps=max_steps),
                "steps": [],
            }

            for step_idx, step in enumerate(steps):
                row["steps"].append(
                    {
                        "step_index": step_idx,
                        "observation": getattr(step, "observation", None),
                        "thought": getattr(step, "thought", ""),
                        "action": getattr(step, "action", None),
                        "model_response": getattr(step, "model_response", ""),
                        "reward": float(getattr(step, "reward", 0.0) or 0.0),
                        "done": bool(getattr(step, "done", False)),
                        "info": getattr(step, "info", {}) or {},
                        "chat_completions": getattr(step, "chat_completions", []) or [],
                    }
                )

            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    print(f"Saved full-step traces to: {output_path}")


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    # Fixed eval settings aligned with EcoQA training setup.
    split = "test"
    repeat_n = 8
    n_parallel_agents = 64
    max_steps = 10
    max_prompt_length = 2048
    temperature = 0.6
    top_p = 0.95

    # Keep model selection simple via env var: ECOQA_MODEL_SOURCE=base|ckpt.
    model_source = os.getenv("ECOQA_MODEL_SOURCE", "base").strip().lower()
    if model_source not in {"base", "ckpt"}:
        model_source = "base"

    base_model_path = os.getenv("ECOQA_BASE_MODEL_PATH", "/root/autodl-tmp/models/Qwen3-4B-Instruct-2507")
    ckpt_model_path = os.getenv("ECOQA_CKPT_MODEL_PATH", "/root/autodl-tmp/checkpoints/rllm-agent/ecoqa-4b")
    model_name = base_model_path if model_source == "base" else ckpt_model_path

    base_url = os.getenv("ECOQA_BASE_URL", "http://127.0.0.1:30000/v1")
    tokenizer_model = os.getenv("ECOQA_TOKENIZER_MODEL", "/root/autodl-tmp/models/Qwen3-4B-Instruct-2507")
    api_key = os.getenv("ECOQA_API_KEY", "EMPTY")
    save_steps_jsonl = os.getenv("ECOQA_SAVE_STEPS_JSONL", "").strip()

    print(f"[EcoQA] split={split}, repeat_n={repeat_n}, max_steps={max_steps}, max_prompt_length={max_prompt_length}")
    print(f"[EcoQA] sampling: temperature={temperature}, top_p={top_p}")
    print(f"[EcoQA] model_source={model_source}")
    print(f"[EcoQA] model={model_name}")
    print(f"[EcoQA] base_url={base_url} (expect vLLM with dtype=bfloat16)")

    engine = build_engine(
        model=model_name,
        base_url=base_url,
        api_key=api_key,
        tokenizer_model=tokenizer_model,
        temperature=temperature,
        top_p=top_p,
        n_parallel_agents=n_parallel_agents,
        max_steps=max_steps,
        max_prompt_length=max_prompt_length,
    )

    dataset = load_split(split)
    tasks = dataset.repeat(n=repeat_n)
    results = asyncio.run(engine.execute_tasks(tasks))

    _print_eval_metrics(results)

    if save_steps_jsonl:
        _write_steps_jsonl(path=save_steps_jsonl, trajectories=results, max_steps=max_steps)


if __name__ == "__main__":
    main()
