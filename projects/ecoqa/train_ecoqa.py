import hydra
from rllm.agents.agent import Episode
from rllm.engine.rollout.rollout_engine import ModelOutput
from rllm.trainer.agent_trainer import AgentTrainer
from rllm.workflows.multi_turn_workflow import MultiTurnWorkflow
from rllm.workflows.workflow import TerminationEvent, TerminationReason

from .eco_qa_agent import EcoQAAgent
from .eco_qa_environment import EcoQAEnvironment


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
            correctness_reward = float(metadata.get("correctness_reward", float(step_info.get("is_correct", False))))

            episode.metrics["correctness_reward"] = correctness_reward
            episode.metrics["final_reward"] = float(
                metadata.get("final_reward", getattr(episode.trajectories[0], "reward", correctness_reward))
            )
            episode.metrics["shaping_bonus"] = float(metadata.get("shaping_bonus", 0.0))
            episode.metrics["exp_table_hit_rate"] = float(metadata.get("exp_table_hit_rate", 0.0))
            episode.metrics["exp_table_sql_succ_rate"] = float(metadata.get("exp_table_sql_succ_rate", 0.0))


@hydra.main(
    config_path="pkg://rllm.trainer.config",
    config_name="agent_ppo_trainer",
    version_base=None,
)
def main(config):
    from .prepare_ecoqa_data import prepare_ecoqa_data

    # Always refresh from QA CSV to avoid stale registry schema.
    train_dataset, val_dataset, _ = prepare_ecoqa_data()

    config.rllm.workflow.use_workflow = True

    max_steps = int(getattr(config.rllm.agent, "max_steps", 12))

    trainer = AgentTrainer(
        workflow_class=EcoQAWorkflow,
        workflow_args={
            "agent_cls": EcoQAAgent,
            "env_cls": EcoQAEnvironment,
            "max_steps": max_steps,
        },
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
