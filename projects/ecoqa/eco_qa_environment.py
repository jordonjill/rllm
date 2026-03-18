import json

from rllm.environments.tools.tool_env import ToolEnvironment
from rllm.rewards.reward_types import RewardOutput

from .eco_qa_reward import eco_qa_reward_function
from .eco_qa_tools import Calculator, GetTableInfo, GetTableNames, SQLQuery


class EcoQAEnvironment(ToolEnvironment):
    """Tool environment for EcoQA."""

    def __init__(self, task: dict | None = None, max_steps: int = 10):
        tool_map = {
            "calculator": Calculator,
            "get_table_info": GetTableInfo,
            "get_table_names": GetTableNames,
            "sql_query": SQLQuery,
        }
        super().__init__(task=task, tool_map=tool_map, reward_fn=eco_qa_reward_function, max_steps=max_steps)
        self.sql_call_records: list[dict[str, object]] = []

    @staticmethod
    def _has_finish_tool_call(action: list[dict]) -> bool:
        for tool_call in action:
            if tool_call.get("function", {}).get("name") == "finish":
                return True
        return False

    def _force_fail_on_step_budget(self, action: list[dict] | dict) -> tuple[dict, float, bool, dict]:
        # Keep max-steps semantics strict: tool calls on the budget-exhausted step
        # are treated as failure instead of being parsed as FINAL ANSWER text.
        task_info = self.task if self.task is not None else {}
        question_type = str(task_info.get("question_type", "")).strip().lower()
        answer_type = str(task_info.get("answer_type", "")).strip().lower()
        if question_type == "single_table_error":
            target_kind = "no_data"
        elif answer_type == "structure":
            target_kind = "structure"
        else:
            target_kind = "unknown"
        reward_output = RewardOutput(
            reward=0.0,
            is_correct=False,
            metadata={
                "correctness_reward": 0.0,
                "shaping_bonus": 0.0,
                "final_reward": 0.0,
                "sql_call_count": len(self.sql_call_records),
                "exp_table_sql_calls": 0,
                "exp_table_sql_success": 0,
                "exp_table_hit_rate": 0.0,
                "exp_table_sql_succ_rate": 0.0,
                "target_kind": target_kind,
                "structure_exact_match": False,
                "structure_alias_value_match": False,
                "pred_structure_valid": 0.0,
                "final_answer_extracted": "",
                "forced_termination_reason": "max_steps_without_final_answer",
                "forced_max_steps_failure": 1.0,
            },
        )
        self._sync_task_runtime_stats()
        return {}, reward_output.reward, True, {"response": action, "metadata": reward_output.metadata, "is_correct": reward_output.is_correct}

    def step(self, action: list[dict] | str | dict):
        normalized_action = action
        if normalized_action is None:
            normalized_action = []
        if isinstance(normalized_action, dict):
            normalized_action = [normalized_action]

        reaches_step_budget = self.step_count + 1 >= self.max_steps
        if reaches_step_budget and isinstance(normalized_action, list) and not self._has_finish_tool_call(normalized_action):
            self.step_count += 1
            return self._force_fail_on_step_budget(normalized_action)

        return super().step(action)

    def reset(self, task: dict | None = None):
        if task is not None:
            self.task = task

        if hasattr(self, "task") and self.task is not None:
            self.task.pop("sql_call_records", None)

        self.sql_call_records = []
        self._sync_task_runtime_stats()
        return super().reset()

    def _execute_tool_calls(self, tool_calls: list[dict]) -> dict[str, str]:
        sql_call_info: dict[str, str] = {}

        for tool_call in tool_calls:
            tool_name = tool_call.get("function", {}).get("name")
            if tool_name not in {"get_table_info", "sql_query"}:
                continue

            try:
                tool_args = json.loads(tool_call["function"]["arguments"])
            except (json.JSONDecodeError, KeyError, TypeError):
                continue

            table_name = str(tool_args.get("table_name", "")).strip().lower()
            if tool_name == "sql_query":
                call_id = str(tool_call.get("id", "")).strip()
                if call_id:
                    sql_call_info[call_id] = table_name

        tool_outputs = super()._execute_tool_calls(tool_calls)

        for call_id, table_name in sql_call_info.items():
            output = str(tool_outputs.get(call_id, "")).strip()
            is_success = not output.lower().startswith("error:")
            self.sql_call_records.append({"table_name": table_name, "success": is_success})

        self._sync_task_runtime_stats()
        return tool_outputs

    def _sync_task_runtime_stats(self) -> None:
        if self.task is None:
            return
        self.task["sql_call_records"] = self.sql_call_records

    @staticmethod
    def from_dict(env_args: dict) -> "EcoQAEnvironment":
        if "task" in env_args:
            task = dict(env_args["task"])
            max_steps = int(env_args.get("max_steps", 10))
        else:
            max_steps = int(env_args.get("max_steps", 10))
            task = {k: v for k, v in env_args.items() if k != "max_steps"}
        return EcoQAEnvironment(task=task, max_steps=max_steps)
