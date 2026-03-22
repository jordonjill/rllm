import pytest

from projects.ecoqa.eco_qa_reward import eco_qa_reward_function


def _task(ground_truth: str):
    return {
        "question": "dummy",
        "ground_truth": ground_truth,
        "table_name": "interest_rates",
    }


def test_exact_match_gets_f1_one():
    task = _task('{"rows":[{"result":8.10725}]}')
    pred = eco_qa_reward_function(task, 'FINAL ANSWER: {"rows":[{"result":8.10725}]}')
    assert pred.reward == 1.0
    assert pred.is_correct is True
    assert pred.metadata["final_reward"] == 1.0
    assert pred.metadata["correctness_reward"] == 1.0
    assert pred.metadata["exp_table_hit_rate"] == 0.0
    assert pred.metadata["exp_table_sql_succ_rate"] == 0.0


def test_percent_symbol_still_normalizes_to_match():
    task = _task('{"rows":[{"rate":5.5}]}')
    pred = eco_qa_reward_function(task, 'FINAL ANSWER: {"rows":[{"rate":"5.5%"}]}')
    assert pred.reward == 1.0
    assert pred.is_correct is True


def test_json_code_block_without_prefix_is_supported():
    task = _task('{"rows":[{"rate":5.5}]}')
    pred = eco_qa_reward_function(task, '```json\n{"rows":[{"rate":"5.5%"}]}\n```')
    assert pred.reward == 1.0


def test_order_insensitive_rows_can_match_exactly():
    task = _task('{"rows":[{"year":2024,"month":1,"a":1},{"b":"2"}]}')
    pred = eco_qa_reward_function(task, 'FINAL ANSWER: {"rows":[{"b":2.0},{"month":1,"a":"1","year":"2024"}]}')
    assert pred.reward == 1.0
    assert pred.is_correct is True


def test_partial_overlap_returns_fractional_f1():
    task = _task('{"rows":[{"result":1},{"result":2}]}')
    pred = eco_qa_reward_function(task, 'FINAL ANSWER: {"rows":[{"result":1}]}')
    assert pred.is_correct is False
    assert pred.reward == pytest.approx(2.0 / 3.0, rel=1e-9)
    assert pred.metadata["final_reward"] == pytest.approx(2.0 / 3.0, rel=1e-9)
    assert pred.metadata["correctness_reward"] == 0.0


def test_extra_wrong_row_lowers_precision_and_f1():
    task = _task('{"rows":[{"result":1}]}')
    pred = eco_qa_reward_function(task, 'FINAL ANSWER: {"rows":[{"result":1},{"result":2}]}')
    assert pred.is_correct is False
    assert pred.reward == pytest.approx(2.0 / 3.0, rel=1e-9)


def test_empty_rows_exact_match_is_one():
    task = _task('{"rows":[]}')
    pred = eco_qa_reward_function(task, 'FINAL ANSWER: {"rows":[]}')
    assert pred.reward == 1.0
    assert pred.is_correct is True


def test_invalid_schema_gets_zero_f1():
    task = _task('{"rows":[{"result":1}]}')
    pred = eco_qa_reward_function(task, 'FINAL ANSWER: {"items":[{"name":"x","value":1}]}')
    assert pred.reward == 0.0
    assert pred.is_correct is False


def test_non_json_prediction_gets_zero_f1():
    task = _task('{"rows":[{"result":1}]}')
    pred = eco_qa_reward_function(task, "FINAL ANSWER: 1")
    assert pred.reward == 0.0
    assert pred.is_correct is False


def test_duplicate_rows_counted_with_multiplicity():
    task = _task('{"rows":[{"result":1},{"result":1}]}')
    pred = eco_qa_reward_function(task, 'FINAL ANSWER: {"rows":[{"result":1}]}')
    assert pred.reward == pytest.approx(2.0 / 3.0, rel=1e-9)
    assert pred.is_correct is False


def test_partial_dict_match_gets_fractional_reward():
    task = _task('{"rows":[{"year":2024,"result":100}]}')
    pred = eco_qa_reward_function(task, 'FINAL ANSWER: {"rows":[{"year":2024,"result":90}]}')
    assert pred.reward == pytest.approx(0.5, rel=1e-9)
    assert pred.is_correct is False


def test_expected_table_hit_and_sql_success_rates_are_reported():
    task = _task('{"rows":[{"result":1}]}')
    task["sql_call_records"] = [
        {"table_name": "interest_rates", "success": True},
        {"table_name": "interest_rates", "success": False},
        {"table_name": "other_table", "success": True},
    ]
    pred = eco_qa_reward_function(task, 'FINAL ANSWER: {"rows":[{"result":1}]}')
    assert pred.metadata["exp_table_hit_rate"] == pytest.approx(2 / 3, rel=1e-9)
    assert pred.metadata["exp_table_sql_succ_rate"] == pytest.approx(1 / 2, rel=1e-9)


def test_expected_table_hit_supports_csv_suffix():
    task = _task('{"rows":[{"result":1}]}')
    task["sql_call_records"] = [{"table_name": "interest_rates.csv", "success": True}]
    pred = eco_qa_reward_function(task, 'FINAL ANSWER: {"rows":[{"result":1}]}')
    assert pred.metadata["exp_table_hit_rate"] == 1.0
    assert pred.metadata["exp_table_sql_succ_rate"] == 1.0
