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
        "ground_truth": '{"rows":[{"x":1}]}',
        "table_name": "interest_rates",
    }
    env = EcoQAEnvironment(task=task)
    env.max_steps = 1
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
    assert set(info["metadata"].keys()) == {
        "final_reward",
        "correctness_reward",
        "exp_table_hit_rate",
        "exp_table_sql_succ_rate",
    }
    assert info["metadata"]["final_reward"] == 0.0
    assert info["metadata"]["correctness_reward"] == 0.0
    assert info["metadata"]["exp_table_hit_rate"] == 0.0
    assert info["metadata"]["exp_table_sql_succ_rate"] == 0.0
    assert task["sql_call_records"] == []
