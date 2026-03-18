import json
import random
import re

import hydra

from rllm.agents.agent import Episode
from rllm.data.dataset import DatasetRegistry
from rllm.engine.rollout.rollout_engine import ModelOutput
from rllm.trainer.agent_trainer import AgentTrainer
from rllm.workflows.multi_turn_workflow import MultiTurnWorkflow
from rllm.workflows.workflow import TerminationEvent, TerminationReason

from .eco_qa_agent import EcoQAAgent
from .eco_qa_environment import EcoQAEnvironment


def _as_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off", ""}:
            return False
    return default


def _infer_target_kind_from_task(task: dict) -> str:
    question_type = str(task.get("question_type", "")).strip().lower()
    answer_type = str(task.get("answer_type", "")).strip().lower()
    if question_type == "single_table_error":
        return "no_data"
    if answer_type == "structure":
        return "structure"
    return "unknown"


def _infer_sql_difficulty(example: dict) -> str:
    question_type = str(example.get("question_type", "")).strip().lower()
    if question_type == "single_table_error":
        return "easy"

    sql = str(example.get("ground_truth_sql", "")).strip().lower()
    answer_type = str(example.get("answer_type", "")).strip().lower()
    ground_truth = str(example.get("ground_truth", "")).strip()
    score = 0.0

    # Heuristic complexity score from SQL operators.
    sql_patterns = (
        r"\bgroup\s+by\b",
        r"\bhaving\b",
        r"\border\s+by\b",
        r"\bcount\s*\(",
        r"\bsum\s*\(",
        r"\bavg\s*\(",
        r"\bmin\s*\(",
        r"\bmax\s*\(",
        r"\bdistinct\b",
        r"\bcase\b",
    )
    for pattern in sql_patterns:
        if re.search(pattern, sql):
            score += 1

    if " and " in sql or " or " in sql:
        score += 1

    # Multi-SQL + calculator questions are harder even if each SQL is simple.
    if _as_bool(example.get("requires_calculator", False), default=False):
        score += 1.5

    if answer_type == "list":
        score += 0.5
    elif answer_type == "structure":
        try:
            payload = json.loads(ground_truth) if ground_truth else {}
            items = payload.get("items", []) if isinstance(payload, dict) else []
        except json.JSONDecodeError:
            items = []

        # Structured answers with multiple items or dims are typically harder.
        if isinstance(items, list):
            if len(items) > 1:
                score += 0.5
            if any(isinstance(it, dict) and isinstance(it.get("dims"), dict) and it.get("dims") for it in items):
                score += 0.5

    if score <= 1:
        return "easy"
    if score <= 3:
        return "medium"
    return "hard"


def _build_curriculum_train_dataset(config, train_dataset):
    curriculum_cfg = config.get("ecoqa_curriculum", None)
    if not curriculum_cfg or not bool(curriculum_cfg.get("enable", False)):
        return train_dataset

    raw_data = train_dataset.get_data()
    if not raw_data:
        return train_dataset

    phase = min(1.0, max(0.0, _as_float(curriculum_cfg.get("phase", 0.35), 0.35)))
    raw_size_multiplier = _as_float(curriculum_cfg.get("size_multiplier", 1.2), 1.2)
    size_multiplier = raw_size_multiplier
    if size_multiplier <= 1.0:
        size_multiplier = 1.2
    seed = _as_int(curriculum_cfg.get("seed", 42), 42)

    difficulty_weight_start = curriculum_cfg.get("difficulty_weight_start", {})
    difficulty_weight_end = curriculum_cfg.get("difficulty_weight_end", {})

    def weight_from_map(weight_cfg, diff_key: str) -> float:
        return _as_float(weight_cfg.get(diff_key, 1.0), 1.0)

    rng = random.Random(seed)
    base_size = len(raw_data)
    target_size = max(base_size + 1, int(round(base_size * size_multiplier)))
    sampled_data = []

    stage1_ratio = phase
    stage1_size = int(round(target_size * stage1_ratio))
    stage2_size = target_size - stage1_size

    start_weights = []
    end_weights = []
    for example in raw_data:
        diff_key = _infer_sql_difficulty(example)
        start_weights.append(max(weight_from_map(difficulty_weight_start, diff_key), 1e-6))
        end_weights.append(max(weight_from_map(difficulty_weight_end, diff_key), 1e-6))

    stage1_indices = rng.choices(range(base_size), weights=start_weights, k=stage1_size) if stage1_size > 0 else []
    stage2_indices = rng.choices(range(base_size), weights=end_weights, k=stage2_size) if stage2_size > 0 else []

    sampled_data.extend(dict(raw_data[idx]) for idx in stage1_indices)
    sampled_data.extend(dict(raw_data[idx]) for idx in stage2_indices)

    print(
        "[EcoQACurriculum] enabled, "
        f"phase={phase:.3f}, size_multiplier={size_multiplier:.3f}, "
        f"base_size={base_size}, sampled_size={len(sampled_data)}, "
        f"stage1_size={stage1_size}, stage2_size={stage2_size}, "
        f"dataset={str(curriculum_cfg.get('dataset_name', 'ecoqa_curriculum')).strip() or 'ecoqa_curriculum'}/{str(curriculum_cfg.get('dataset_split', 'train')).strip() or 'train'}"
    )
    if raw_size_multiplier <= 1.0:
        print(
            "[EcoQACurriculum] size_multiplier<=1 detected; "
            f"auto-adjusted from {raw_size_multiplier:.3f} to {size_multiplier:.3f} to ensure extra curriculum samples."
        )

    dataset_name = str(curriculum_cfg.get("dataset_name", "ecoqa_curriculum")).strip() or "ecoqa_curriculum"
    dataset_split = str(curriculum_cfg.get("dataset_split", "train")).strip() or "train"
    sampled_dataset = DatasetRegistry.register_dataset(dataset_name, sampled_data, dataset_split)
    return sampled_dataset


class EcoQAWorkflow(MultiTurnWorkflow):
    """MultiTurnWorkflow with reward metadata logging."""

    async def run(self, task: dict, uid: str, **kwargs) -> Episode | None:
        observation, info = await self.run_in_executor(self.reset, task=task, uid=uid)

        self.agent.update_from_env(observation, 0, False, info)

        max_model_len = self.rollout_engine.max_prompt_length + self.rollout_engine.max_response_length
        min_response_buffer = 1000

        for _ in range(1, self.max_steps + 1):
            if hasattr(self.rollout_engine, "chat_parser"):
                prompt = self.rollout_engine.chat_parser.parse(
                    self.agent.chat_completions,
                    add_generation_prompt=True,
                    is_first_msg=True,
                )
                prompt_length = len(self.rollout_engine.tokenizer.encode(prompt, add_special_tokens=False))
            else:
                prompt_ids = self.rollout_engine.tokenizer.apply_chat_template(
                    self.agent.chat_completions,
                    add_generation_prompt=True,
                    tokenize=True,
                )
                prompt_length = len(prompt_ids)

            if prompt_length > max_model_len - min_response_buffer:
                raise TerminationEvent(TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED)

            output: ModelOutput = await self.rollout_engine.get_model_response(
                self.agent.chat_completions,
                application_id=uid,
                enforce_max_prompt_length=False,
                **kwargs,
            )
            response = output.text

            action = self.agent.update_from_model(response)

            if not hasattr(self.rollout_engine, "chat_parser") and self.agent.trajectory.steps:
                self.agent.trajectory.steps[-1].model_output = output

            next_obs, reward, done, info = await self.run_in_executor(self.env.step, action.action)
            self.agent.update_from_env(next_obs, reward, done, info)

            if output.finish_reason == "length":
                raise TerminationEvent(TerminationReason.MAX_RESPONSE_LENGTH_EXCEEDED)

            if done:
                raise TerminationEvent(TerminationReason.ENV_DONE)

        raise TerminationEvent(TerminationReason.MAX_TURNS_EXCEEDED)

    def assign_episode_correctness(self, episode: Episode) -> None:
        if episode.trajectories and episode.trajectories[0].steps:
            is_correct = episode.trajectories[0].steps[-1].info.get("is_correct")
            if is_correct is not None:
                episode.is_correct = is_correct
                return
        super().assign_episode_correctness(episode)

    def collect_metrics(self, episode: Episode) -> None:
        super().collect_metrics(episode)
        if episode.trajectories and episode.trajectories[0].steps:
            step_info = episode.trajectories[0].steps[-1].info
            metadata = step_info.get("metadata", {})
            target_kind = str(metadata.get("target_kind", "")).strip().lower()
            if not target_kind:
                task = episode.task if isinstance(episode.task, dict) else {}
                target_kind = _infer_target_kind_from_task(task)
            correctness_reward = float(metadata.get("correctness_reward", float(step_info.get("is_correct", False))))

            # Keep overall rewards for global monitoring.
            episode.metrics["correctness_reward"] = correctness_reward
            episode.metrics["final_reward"] = float(metadata.get("final_reward", correctness_reward))
            episode.metrics["shaping_bonus"] = float(metadata.get("shaping_bonus", 0.0))
            episode.metrics["exp_table_hit_rate"] = float(metadata.get("exp_table_hit_rate", 0.0))
            episode.metrics["exp_table_sql_succ_rate"] = float(metadata.get("exp_table_sql_succ_rate", 0.0))
            episode.metrics["forced_max_steps_failure"] = float(metadata.get("forced_max_steps_failure", 0.0))
            episode.metrics["pred_structure_valid"] = float(metadata.get("pred_structure_valid", 0.0))

            # Dataset composition monitoring.
            episode.metrics["target_is_structure"] = 1.0 if target_kind == "structure" else 0.0
            episode.metrics["target_is_no_data"] = 1.0 if target_kind == "no_data" else 0.0
            episode.metrics["target_is_unknown"] = 1.0 if target_kind not in {"structure", "no_data"} else 0.0

            # Type-conditional accuracy (denominator = samples of that type only).
            if target_kind == "structure":
                episode.metrics["structure_acc"] = correctness_reward
                episode.metrics["structure_exact_match"] = float(bool(metadata.get("structure_exact_match", False)))
                episode.metrics["structure_alias_value_match"] = float(bool(metadata.get("structure_alias_value_match", False)))
            elif target_kind == "no_data":
                episode.metrics["no_data_acc"] = correctness_reward


@hydra.main(
    config_path="pkg://rllm.trainer.config",
    config_name="agent_ppo_trainer",
    version_base=None,
)
def main(config):
    train_dataset = DatasetRegistry.load_dataset("ecoqa", "train")
    val_dataset = DatasetRegistry.load_dataset("ecoqa", "val")

    if train_dataset is None or val_dataset is None:
        from .prepare_ecoqa_data import prepare_ecoqa_data

        train_dataset, val_dataset, _ = prepare_ecoqa_data()

    train_dataset = _build_curriculum_train_dataset(config, train_dataset)

    curriculum_cfg = config.get("ecoqa_curriculum", None)
    if curriculum_cfg and bool(curriculum_cfg.get("enable", False)):
        data_cfg = config.get("data", {})
        if bool(data_cfg.get("shuffle", True)):
            print("[EcoQACurriculum] warning: data.shuffle=True will break stage order; set data.shuffle=False.")

    config.rllm.workflow.use_workflow = True

    max_steps = int(getattr(config.rllm.agent, "max_steps", 12))

    trainer = AgentTrainer(
        workflow_class=EcoQAWorkflow,
        workflow_args={
            "agent_cls": EcoQAAgent,
            "env_cls": EcoQAEnvironment,
            "env_args": {"max_steps": max_steps},
            "max_steps": max_steps,
        },
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
