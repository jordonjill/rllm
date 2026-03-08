from rllm.agents.tool_agent import ToolAgent

from .constants import REACT_SYSTEM_PROMPT_PATH
from .eco_qa_tools import Calculator, GetTableInfo, GetTableNames, SQLQuery

with open(REACT_SYSTEM_PROMPT_PATH, encoding="utf-8") as f:
    ECOQA_REACT_SYSTEM_PROMPT = f.read().strip()


class EcoQAAgent(ToolAgent):
    """Single-agent tool caller for single-table numerical QA."""

    def __init__(
        self,
        system_prompt: str = ECOQA_REACT_SYSTEM_PROMPT,
        parser_name: str = "qwen",
    ):
        ecoqa_tool_map = {
            "calculator": Calculator,
            "get_table_info": GetTableInfo,
            "get_table_names": GetTableNames,
            "sql_query": SQLQuery,
        }

        super().__init__(
            system_prompt=system_prompt,
            parser_name=parser_name,
            tool_map=ecoqa_tool_map,
        )
