import asyncio
import argparse
import hashlib
import json
import os
from pathlib import Path

from transformers import AutoTokenizer

from rllm.data.dataset import DatasetRegistry
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.utils import compute_pass_at_k

from .eco_qa_agent import EcoQAAgent
from .eco_qa_environment import EcoQAEnvironment


def _task_id(task: dict) -> str:
    qid = str(task.get("question_id", "")).strip()
    if qid:
        return qid
    digest = hashlib.md5(json.dumps(task, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    return digest


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
                "question_id": _task_id(task),
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
    parser = argparse.ArgumentParser(description="Run full EcoQA inference-eval pipeline for one model.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--repeat-n", type=int, default=1, help="Repeat each sample N times for pass@k.")
    parser.add_argument("--max-samples", type=int, default=0, help="Use first N samples for quick checks; 0 means all.")
    parser.add_argument("--n-parallel-agents", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--max-prompt-length", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--model", type=str, default=os.getenv("ECOQA_MODEL", "qwen3-4b:latest"))
    parser.add_argument("--base-url", type=str, default=os.getenv("ECOQA_BASE_URL", "http://localhost:11434/v1"))
    parser.add_argument(
        "--tokenizer-model",
        type=str,
        default=os.getenv("ECOQA_TOKENIZER_MODEL", "Qwen/Qwen3-4B-Instruct-2507"),
        help="HF tokenizer repo/path. Keep this separate from serving model name (e.g. Ollama tag).",
    )
    parser.add_argument("--api-key", type=str, default=os.getenv("ECOQA_API_KEY", "ollama"))
    parser.add_argument(
        "--save-steps-jsonl",
        type=str,
        default="",
        help="Optional path to save full per-trajectory step traces as JSONL.",
    )
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    model_name = args.model
    base_url = args.base_url

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model)
    except Exception as e:
        raise RuntimeError(
            "Failed to load tokenizer. If you are using Ollama, set --tokenizer-model to a valid "
            "HuggingFace tokenizer repo/path (for qwen3-4b, try Qwen/Qwen3-4B-Instruct-2507)."
        ) from e

    sampling_params = {"temperature": args.temperature, "top_p": args.top_p}

    engine = AgentExecutionEngine(
        agent_class=EcoQAAgent,
        env_class=EcoQAEnvironment,
        engine_name="openai",
        rollout_engine_args={"model": model_name, "base_url": base_url, "api_key": args.api_key},
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        n_parallel_agents=args.n_parallel_agents,
        max_steps=args.max_steps,
        max_prompt_length=args.max_prompt_length,
    )

    dataset = DatasetRegistry.load_dataset("ecoqa", args.split)
    if dataset is None:
        print("EcoQA dataset not found, preparing dataset...")
        from .prepare_ecoqa_data import prepare_ecoqa_data

        train_dataset, val_dataset, test_dataset = prepare_ecoqa_data()
        dataset = {"train": train_dataset, "val": val_dataset, "test": test_dataset}[args.split]

    if args.max_samples > 0:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    tasks = dataset.repeat(n=args.repeat_n)
    results = asyncio.run(engine.execute_tasks(tasks))

    compute_pass_at_k(results)

    if args.save_steps_jsonl:
        _write_steps_jsonl(path=args.save_steps_jsonl, trajectories=results, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
