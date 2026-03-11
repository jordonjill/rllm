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
        self.sql_total_calls = 0
        self.sql_success_calls = 0
        self.sql_error_calls = 0

    def reset(self, task: dict | None = None):
        if task is not None:
            self.task = task

        if hasattr(self, "task") and self.task is not None:
            self.task.pop("accessed_tables", None)
            self.task.pop("sql_total_calls", None)
            self.task.pop("sql_success_calls", None)
            self.task.pop("sql_error_calls", None)

        self.accessed_tables = []
        self.sql_total_calls = 0
        self.sql_success_calls = 0
        self.sql_error_calls = 0
        self._sync_task_runtime_stats()
        return super().reset()

    def _execute_tool_calls(self, tool_calls: list[dict]) -> dict[str, str]:
        sql_call_ids: list[str] = []

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

            if tool_name == "sql_query":
                call_id = str(tool_call.get("id", "")).strip()
                if call_id:
                    sql_call_ids.append(call_id)

        tool_outputs = super()._execute_tool_calls(tool_calls)

        for call_id in sql_call_ids:
            self.sql_total_calls += 1
            output = str(tool_outputs.get(call_id, "")).strip()
            if output.lower().startswith("error:"):
                self.sql_error_calls += 1
            else:
                self.sql_success_calls += 1

        self._sync_task_runtime_stats()
        return tool_outputs

    def _sync_task_runtime_stats(self) -> None:
        if self.task is None:
            return
        self.task["accessed_tables"] = self.accessed_tables
        self.task["sql_total_calls"] = self.sql_total_calls
        self.task["sql_success_calls"] = self.sql_success_calls
        self.task["sql_error_calls"] = self.sql_error_calls

    @staticmethod
    def from_dict(env_args: dict) -> "EcoQAEnvironment":
        task = env_args["task"] if "task" in env_args else env_args
        return EcoQAEnvironment(task=task)
