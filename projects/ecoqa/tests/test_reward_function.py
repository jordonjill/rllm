from projects.ecoqa.eco_qa_reward import eco_qa_reward_function


def _task(ground_truth: str, question_type: str, answer_type: str):
    return {
        "question": "dummy",
        "ground_truth": ground_truth,
        "question_type": question_type,
        "answer_type": answer_type,
        "table_name": "interest_rates",
        "accessed_tables": ["interest_rates"],
    }


def test_scalar_reward_binary():
    task = _task("8.10725", "single_table", "scalar")
    correct = eco_qa_reward_function(task, 'FINAL ANSWER: {"type":"scalar","value":8.10725}')
    wrong = eco_qa_reward_function(task, 'FINAL ANSWER: {"type":"scalar","value":7.0}')
    assert correct.reward == 1.0 and correct.is_correct
    assert wrong.reward == 0.0 and not wrong.is_correct


def test_list_requires_exact_match():
    gt = '[{"a":1,"b":2},{"a":3,"b":4}]'
    task = _task(gt, "single_table", "list")
    correct = eco_qa_reward_function(task, 'FINAL ANSWER: {"type":"list","rows":[{"b":"2","a":1.0},{"a":3,"b":4}]}')
    wrong = eco_qa_reward_function(task, 'FINAL ANSWER: {"type":"list","rows":[{"a":1,"b":2}]}')
    assert correct.reward == 1.0 and correct.is_correct
    assert wrong.reward == 0.0 and not wrong.is_correct


def test_error_semantic_match_binary():
    task = _task("数据范围仅覆盖2016-2025年，无法查询2030年数据", "single_table_error", "")
    correct = eco_qa_reward_function(task, 'FINAL ANSWER: {"type":"error","reason":"无数据"}')
    wrong = eco_qa_reward_function(task, 'FINAL ANSWER: {"type":"error","reason":"2024年为8.1%"}')
    assert correct.reward == 1.0 and correct.is_correct
    assert wrong.reward == 0.0 and not wrong.is_correct
