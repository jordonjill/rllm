import json

from rllm.environments.tools.tool_env import ToolEnvironment

from .eco_qa_reward import eco_qa_reward_function
from .eco_qa_tools import Calculator, GetTableInfo, GetTableNames, SQLQuery


class EcoQAEnvironment(ToolEnvironment):
    """Tool environment for EcoQA."""

    def __init__(self, task: dict | None = None):
        tool_map = {
            "calculator": Calculator,
            "get_table_info": GetTableInfo,
            "get_table_names": GetTableNames,
            "sql_query": SQLQuery,
        }
        super().__init__(task=task, tool_map=tool_map, reward_fn=eco_qa_reward_function, max_steps=12)
        self.accessed_tables: list[str] = []

    def reset(self, task: dict | None = None):
        if task is not None:
            self.task = task

        if hasattr(self, "task") and self.task is not None:
            self.task.pop("accessed_tables", None)

        self.accessed_tables = []
        return super().reset()

    def _execute_tool_calls(self, tool_calls: list[dict]) -> dict[str, str]:
        for tool_call in tool_calls:
            tool_name = tool_call.get("function", {}).get("name")
            if tool_name not in {"get_table_info", "sql_query"}:
                continue

            try:
                tool_args = json.loads(tool_call["function"]["arguments"])
            except (json.JSONDecodeError, KeyError, TypeError):
                continue

            table_name = str(tool_args.get("table_name", "")).strip().lower()
            if table_name:
                self.accessed_tables.append(table_name)

        if self.task is not None:
            self.task["accessed_tables"] = self.accessed_tables

        return super()._execute_tool_calls(tool_calls)

    @staticmethod
    def from_dict(env_args: dict) -> "EcoQAEnvironment":
        task = env_args["task"] if "task" in env_args else env_args
        return EcoQAEnvironment(task=task)
