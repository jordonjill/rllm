import asyncio
import argparse
import os

from transformers import AutoTokenizer

from rllm.data.dataset import DatasetRegistry
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.utils import compute_pass_at_k

from .eco_qa_agent import EcoQAAgent
from .eco_qa_environment import EcoQAEnvironment


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
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    # model_name = os.getenv("ECOQA_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
    # base_url = os.getenv("ECOQA_BASE_URL", "http://localhost:30000/v1")
    model_name = os.getenv("ECOQA_MODEL", "qwen3-4b")
    base_url = os.getenv("ECOQA_BASE_URL", "http://localhost:11434")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    sampling_params = {"temperature": args.temperature, "top_p": args.top_p}

    engine = AgentExecutionEngine(
        agent_class=EcoQAAgent,
        env_class=EcoQAEnvironment,
        engine_name="openai",
        rollout_engine_args={"model": model_name, "base_url": base_url, "api_key": "None"},
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


if __name__ == "__main__":
    main()
