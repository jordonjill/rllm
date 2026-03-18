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

    assert task["sql_call_records"] == [
        {"table_name": "interest_rates", "success": True},
        {"table_name": "interest_rates", "success": False},
    ]


def test_environment_forces_failure_on_step_budget_tool_call():
    task = {
        "question": "dummy",
        "ground_truth": '{"items":[{"name":"x","value":1}]}',
        "question_type": "single_table",
        "answer_type": "structure",
        "table_name": "interest_rates",
    }
    env = EcoQAEnvironment(task=task, max_steps=1)
    env.reset()

    obs, reward, done, info = env.step(
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

    assert done is True
    assert reward == 0.0
    assert obs == {}
    assert info["is_correct"] is False
    assert info["metadata"]["forced_termination_reason"] == "max_steps_without_final_answer"
    assert task["sql_call_records"] == []
