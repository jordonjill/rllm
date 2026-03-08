import asyncio
import os

from transformers import AutoTokenizer

from rllm.data.dataset import DatasetRegistry
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.utils import compute_pass_at_k

from .eco_qa_agent import EcoQAAgent
from .eco_qa_environment import EcoQAEnvironment


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    model_name = os.getenv("ECOQA_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
    base_url = os.getenv("ECOQA_BASE_URL", "http://localhost:30000/v1")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    sampling_params = {"temperature": 0.6, "top_p": 0.95}

    engine = AgentExecutionEngine(
        agent_class=EcoQAAgent,
        env_class=EcoQAEnvironment,
        engine_name="openai",
        rollout_engine_args={"model": model_name, "base_url": base_url, "api_key": "None"},
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        n_parallel_agents=64,
        max_steps=12,
        max_prompt_length=4096,
    )

    test_dataset = DatasetRegistry.load_dataset("ecoqa", "test")
    if test_dataset is None:
        print("EcoQA dataset not found, preparing dataset...")
        from .prepare_ecoqa_data import prepare_ecoqa_data

        _, _, test_dataset = prepare_ecoqa_data()

    tasks = test_dataset.repeat(n=1)
    results = asyncio.run(engine.execute_tasks(tasks))

    compute_pass_at_k(results)
