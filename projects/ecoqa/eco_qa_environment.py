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
        super().__init__(task=task, tool_map=tool_map, reward_fn=eco_qa_reward_function)
        self.sql_call_records: list[dict[str, object]] = []

    def reset(self, task: dict | None = None):
        if task is not None:
            self.task = task

        if self.task is not None:
            self.task.pop("sql_call_records", None)

        self.sql_call_records = []
        if self.task is not None:
            self.task["sql_call_records"] = self.sql_call_records
        return super().reset()

    def _execute_tool_calls(self, tool_calls: list[dict]) -> dict[str, str]:
        sql_call_info: dict[str, str] = {}

        for tool_call in tool_calls:
            tool_name = tool_call.get("function", {}).get("name")
            if tool_name != "sql_query":
                continue

            try:
                tool_args = json.loads(tool_call["function"]["arguments"])
            except (json.JSONDecodeError, KeyError, TypeError):
                continue

            table_name = str(tool_args.get("table_name", "")).strip().lower()
            call_id = str(tool_call.get("id", "")).strip()
            if call_id:
                sql_call_info[call_id] = table_name

        tool_outputs = super()._execute_tool_calls(tool_calls)

        for call_id, table_name in sql_call_info.items():
            output = str(tool_outputs.get(call_id, "")).strip()
            is_success = self._is_valid_sql_json_list(output)
            self.sql_call_records.append({"table_name": table_name, "success": is_success})

        if self.task is not None:
            self.task["sql_call_records"] = self.sql_call_records
        return tool_outputs

    @staticmethod
    def _is_valid_sql_json_list(output: str) -> bool:
        try:
            parsed = json.loads(output)
        except (TypeError, json.JSONDecodeError):
            return False
        return isinstance(parsed, list)

    @staticmethod
    def from_dict(env_args: dict) -> "EcoQAEnvironment":
        if "task" in env_args:
            task = dict(env_args["task"])
        else:
            task = dict(env_args)
        return EcoQAEnvironment(task=task)
