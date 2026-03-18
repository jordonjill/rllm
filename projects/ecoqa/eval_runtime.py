import hashlib
import json
from typing import Any

from transformers import AutoTokenizer

from rllm.data.dataset import DatasetRegistry
from rllm.engine.agent_execution_engine import AgentExecutionEngine

from .eco_qa_agent import EcoQAAgent
from .eco_qa_environment import EcoQAEnvironment
from .prepare_ecoqa_data import prepare_ecoqa_data


def task_id(task: dict[str, Any]) -> str:
    qid = str(task.get("question_id", "")).strip()
    if qid:
        return qid
    digest = hashlib.md5(json.dumps(task, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    return digest


def load_split(split: str):
    dataset = DatasetRegistry.load_dataset("ecoqa", split)
    if dataset is None:
        train, val, test = prepare_ecoqa_data()
        dataset = {"train": train, "val": val, "test": test}[split]
    return dataset


def build_engine(
    *,
    model: str,
    base_url: str,
    api_key: str,
    tokenizer_model: str,
    temperature: float,
    top_p: float,
    n_parallel_agents: int,
    max_steps: int,
    max_prompt_length: int,
) -> AgentExecutionEngine:
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    except Exception as e:
        raise RuntimeError(
            "Failed to load tokenizer. If you are using Ollama, set tokenizer model to a valid "
            "HuggingFace tokenizer repo/path (for qwen3-4b, try Qwen/Qwen3-4B-Instruct-2507)."
        ) from e

    return AgentExecutionEngine(
        agent_class=EcoQAAgent,
        env_class=EcoQAEnvironment,
        env_args={"max_steps": max_steps},
        engine_name="openai",
        rollout_engine_args={"model": model, "base_url": base_url, "api_key": api_key},
        tokenizer=tokenizer,
        sampling_params={"temperature": temperature, "top_p": top_p},
        n_parallel_agents=n_parallel_agents,
        max_steps=max_steps,
        max_prompt_length=max_prompt_length,
    )
