from projects.ecoqa.eco_qa_reward import eco_qa_reward_function


def _task(
    ground_truth: str,
    question_type: str,
    answer_type: str,
    *,
    sql_call_records: list[dict] | None = None,
):
    return {
        "question": "dummy",
        "ground_truth": ground_truth,
        "question_type": question_type,
        "answer_type": answer_type,
        "table_name": "interest_rates",
        "sql_call_records": sql_call_records if sql_call_records is not None else [],
    }


def test_scalar_reward_binary():
    task = _task("8.10725", "single_table", "scalar")
    correct = eco_qa_reward_function(task, 'FINAL ANSWER: {"type":"scalar","value":8.10725}')
    wrong = eco_qa_reward_function(task, 'FINAL ANSWER: {"type":"scalar","value":7.0}')
    assert correct.reward == 1.0 and correct.is_correct
    assert wrong.reward == 0.0 and not wrong.is_correct


def test_scalar_allows_percent_symbol_when_numeric_value_matches():
    task = _task("5.5", "single_table", "scalar")
    pred = eco_qa_reward_function(task, 'FINAL ANSWER: {"type":"scalar","value":"5.5%"}')
    assert pred.reward == 1.0 and pred.is_correct


def test_scalar_text_answer_supported():
    task = _task("广东", "single_table", "scalar")
    pred = eco_qa_reward_function(task, 'FINAL ANSWER: {"type":"scalar","value":"广东"}')
    wrong = eco_qa_reward_function(task, 'FINAL ANSWER: {"type":"scalar","value":"江苏"}')
    assert pred.reward == 1.0 and pred.is_correct
    assert wrong.reward == 0.0 and not wrong.is_correct


def test_list_requires_exact_match():
    gt = '[{"a":1,"b":2},{"a":3,"b":4}]'
    task = _task(gt, "single_table", "list")
    correct = eco_qa_reward_function(task, 'FINAL ANSWER: {"type":"list","rows":[{"b":"2","a":1.0},{"a":3,"b":4}]}')
    wrong = eco_qa_reward_function(task, 'FINAL ANSWER: {"type":"list","rows":[{"a":1,"b":2}]}')
    assert correct.reward == 1.0 and correct.is_correct
    assert wrong.reward == 0.0 and not wrong.is_correct


def test_list_allows_alias_when_row_values_align():
    gt = '[{"geo_name":"华中","avg_export_yoy":7.56}]'
    task = _task(gt, "single_table", "list")
    alias_only = eco_qa_reward_function(
        task,
        'FINAL ANSWER: {"type":"list","rows":[{"geo_name":"华中","avg_export_yoy_pct":"7.56"}]}',
    )
    assert alias_only.reward == 1.0 and alias_only.is_correct
    assert alias_only.metadata.get("list_alias_value_match") is True


def test_list_empty_exact_match_is_correct():
    gt = "[]"
    task = _task(gt, "single_table", "list")
    pred = eco_qa_reward_function(task, 'FINAL ANSWER: {"type":"list","rows":[]}')
    assert pred.reward == 1.0 and pred.is_correct


def test_list_no_partial_reward_for_incomplete_structure():
    gt = '[{"geo_name":"广东","m2_100m_cny":411709.31}]'
    task = _task(gt, "single_table", "list")
    wrong = eco_qa_reward_function(task, 'FINAL ANSWER: {"type":"scalar","value":411709.31}')
    assert wrong.reward == 0.0 and not wrong.is_correct


def test_list_temporal_only_match_is_not_enough_anymore():
    gt = '[{"ref_date":"2017-09-01","wti_usd_per_bbl":61.969}]'
    task = _task(gt, "single_table", "list")
    pred = eco_qa_reward_function(
        task,
        'FINAL ANSWER: {"type":"list","rows":[{"year":2017,"month":9,"min_price":61.969}]}',
    )
    assert pred.reward == 0.0 and not pred.is_correct
    assert "list_temporal_value_match" not in pred.metadata


def test_no_data_requires_no_data_answer():
    task = _task("数据范围仅覆盖2016-2025年，无法查询2030年数据", "single_table_error", "")
    correct = eco_qa_reward_function(task, "FINAL ANSWER: No Data")
    also_correct = eco_qa_reward_function(task, 'FINAL ANSWER: {"type":"no_data"}')
    wrong = eco_qa_reward_function(task, 'FINAL ANSWER: {"type":"error","reason":"逻辑冲突：条件不成立"}')
    assert correct.reward == 1.0 and correct.is_correct
    assert also_correct.reward == 1.0 and also_correct.is_correct
    assert wrong.reward == 0.0 and not wrong.is_correct


def test_incorrect_answer_gets_progress_bonus_when_right_table_and_sql_success():
    task = _task(
        "8.10725",
        "single_table",
        "scalar",
        sql_call_records=[
            {"table_name": "interest_rates", "success": True},
            {"table_name": "interest_rates", "success": True},
        ],
    )
    wrong = eco_qa_reward_function(task, 'FINAL ANSWER: {"type":"scalar","value":7.0}')
    assert wrong.reward > 0.0 and wrong.reward < 1.0
    assert wrong.metadata["exp_table_hit_rate"] == 1.0
    assert wrong.metadata["exp_table_sql_succ_rate"] == 1.0


def test_incorrect_answer_gets_no_bonus_without_sql_success():
    task = _task(
        "8.10725",
        "single_table",
        "scalar",
        sql_call_records=[
            {"table_name": "interest_rates", "success": False},
            {"table_name": "interest_rates", "success": False},
        ],
    )
    wrong = eco_qa_reward_function(task, 'FINAL ANSWER: {"type":"scalar","value":7.0}')
    assert wrong.reward == 0.0
    assert wrong.metadata["exp_table_hit_rate"] == 1.0
    assert wrong.metadata["exp_table_sql_succ_rate"] == 0.0


def test_incorrect_answer_gets_no_bonus_when_only_wrong_table_sql_succeeds():
    task = _task(
        "8.10725",
        "single_table",
        "scalar",
        sql_call_records=[
            {"table_name": "other_table", "success": True},
            {"table_name": "other_table", "success": True},
        ],
    )
    wrong = eco_qa_reward_function(task, 'FINAL ANSWER: {"type":"scalar","value":7.0}')
    assert wrong.reward == 0.0
    assert wrong.metadata["exp_table_hit_rate"] == 0.0
    assert wrong.metadata["exp_table_sql_succ_rate"] == 0.0
