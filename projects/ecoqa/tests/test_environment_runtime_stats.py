import json

from projects.ecoqa.eco_qa_environment import EcoQAEnvironment


def _tool_call(call_id: str, name: str, arguments: dict) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, ensure_ascii=False),
        },
    }


def test_environment_tracks_sql_runtime_stats():
    task = {
        "question": "dummy",
        "ground_truth": "1",
        "question_type": "single_table",
        "answer_type": "scalar",
        "table_name": "interest_rates",
    }
    env = EcoQAEnvironment(task=task)
    env.reset()

    # Successful SQL query.
    env.step(
        [
            _tool_call(
                "1",
                "sql_query",
                {
                    "table_name": "interest_rates",
                    "query": "SELECT year FROM interest_rates LIMIT 1",
                },
            )
        ]
    )

    # SQL error (SELECT * blocked).
    env.step(
        [
            _tool_call(
                "2",
                "sql_query",
                {
                    "table_name": "interest_rates",
                    "query": "SELECT * FROM interest_rates LIMIT 1",
                },
            )
        ]
    )

    assert task["sql_total_calls"] == 2
    assert task["sql_success_calls"] == 1
    assert task["sql_error_calls"] == 1
    assert "interest_rates" in task["accessed_tables"]
