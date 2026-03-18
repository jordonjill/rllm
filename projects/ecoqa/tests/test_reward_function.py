from projects.ecoqa.eco_qa_reward import eco_qa_reward_function


def _task(
    ground_truth: str,
    question_type: str = "single_table",
    answer_type: str = "structure",
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


def test_structure_numeric_value_match():
    task = _task('{"items":[{"name":"result","value":8.10725}]}')
    correct = eco_qa_reward_function(task, 'FINAL ANSWER: {"items":[{"name":"result","value":8.10725}]}')
    wrong = eco_qa_reward_function(task, 'FINAL ANSWER: {"items":[{"name":"result","value":7.0}]}')
    assert correct.reward == 1.0 and correct.is_correct
    assert wrong.reward == 0.0 and not wrong.is_correct


def test_structure_numeric_value_allows_percent_symbol():
    task = _task('{"items":[{"name":"rate","value":5.5}]}')
    pred = eco_qa_reward_function(task, 'FINAL ANSWER: {"items":[{"name":"rate","value":"5.5%"}]}')
    assert pred.reward == 1.0 and pred.is_correct


def test_structure_text_value_match():
    task = _task('{"items":[{"name":"geo_name","value":"广东"}]}')
    pred = eco_qa_reward_function(task, 'FINAL ANSWER: {"items":[{"name":"geo_name","value":"广东"}]}')
    wrong = eco_qa_reward_function(task, 'FINAL ANSWER: {"items":[{"name":"geo_name","value":"江苏"}]}')
    assert pred.reward == 1.0 and pred.is_correct
    assert wrong.reward == 0.0 and not wrong.is_correct


def test_structure_multiple_items_match_order_insensitive():
    gt = '{"items":[{"name":"a","value":1,"dims":{"year":2024,"month":1}},{"name":"b","value":"2"}]}'
    task = _task(gt)
    pred = eco_qa_reward_function(
        task,
        'FINAL ANSWER: {"items":[{"name":"b","value":2.0},{"name":"a","value":"1","dims":{"month":1,"year":"2024"}}]}',
    )
    assert pred.reward == 1.0 and pred.is_correct
    assert pred.metadata.get("structure_exact_match") is True


def test_structure_alias_name_without_dims_is_correct():
    task = _task('{"items":[{"name":"avg_rate","value":3.21}]}')
    pred = eco_qa_reward_function(task, 'FINAL ANSWER: {"items":[{"name":"value","value":"3.21"}]}')
    assert pred.reward == 1.0 and pred.is_correct
    assert pred.metadata.get("structure_exact_match") is False
    assert pred.metadata.get("structure_alias_value_match") is True


def test_structure_alias_name_with_dims_is_correct():
    task = _task('{"items":[{"name":"avg_exports","value":36774.2367,"dims":{"geo_name":"广东","quarter":"Q2"}}]}')
    pred = eco_qa_reward_function(
        task,
        'FINAL ANSWER: {"items":[{"name":"value","value":"36774.2367","dims":{"quarter":"Q2","geo_name":"广东"}}]}',
    )
    assert pred.reward == 1.0 and pred.is_correct
    assert pred.metadata.get("structure_alias_value_match") is True


def test_structure_with_dims_requires_dims_in_prediction():
    task = _task('{"items":[{"name":"avg_exports","value":36774.2367,"dims":{"geo_name":"广东","quarter":"Q2"}}]}')
    wrong = eco_qa_reward_function(task, 'FINAL ANSWER: {"items":[{"name":"value","value":36774.2367}]}')
    assert wrong.reward == 0.0 and not wrong.is_correct


def test_structure_no_partial_reward_for_wrong_schema():
    gt = '{"items":[{"name":"geo_name","value":"广东"},{"name":"m2_100m_cny","value":411709.31}]}'
    task = _task(gt)
    wrong = eco_qa_reward_function(task, 'FINAL ANSWER: {"type":"list","rows":[{"geo_name":"广东"}]}')
    assert wrong.reward == 0.0 and not wrong.is_correct


def test_structure_empty_items_match():
    task = _task('{"items":[]}', question_type="single_table_error")
    pred = eco_qa_reward_function(task, 'FINAL ANSWER: {"items":[]}')
    wrong = eco_qa_reward_function(task, 'FINAL ANSWER: {"items":[{"name":"x","value":1}]}')
    assert pred.reward == 1.0 and pred.is_correct
    assert wrong.reward == 0.0 and not wrong.is_correct


def test_single_table_error_target_kind_is_no_data():
    task = _task('{"items":[]}', question_type="single_table_error")
    pred = eco_qa_reward_function(task, 'FINAL ANSWER: {"items":[]}')
    assert pred.metadata["target_kind"] == "no_data"


def test_incorrect_answer_gets_progress_bonus_when_right_table_and_sql_success():
    task = _task(
        '{"items":[{"name":"result","value":8.10725}]}',
        sql_call_records=[
            {"table_name": "interest_rates", "success": True},
            {"table_name": "interest_rates", "success": True},
        ],
    )
    wrong = eco_qa_reward_function(task, 'FINAL ANSWER: {"items":[{"name":"result","value":7.0}]}')
    assert wrong.reward > 0.0 and wrong.reward < 1.0
    assert wrong.metadata["exp_table_hit_rate"] == 1.0
    assert wrong.metadata["exp_table_sql_succ_rate"] == 1.0


def test_incorrect_answer_gets_no_bonus_without_sql_success():
    task = _task(
        '{"items":[{"name":"result","value":8.10725}]}',
        sql_call_records=[
            {"table_name": "interest_rates", "success": False},
            {"table_name": "interest_rates", "success": False},
        ],
    )
    wrong = eco_qa_reward_function(task, 'FINAL ANSWER: {"items":[{"name":"result","value":7.0}]}')
    assert wrong.reward == 0.0
    assert wrong.metadata["exp_table_hit_rate"] == 1.0
    assert wrong.metadata["exp_table_sql_succ_rate"] == 0.0


def test_incorrect_answer_gets_no_bonus_when_only_wrong_table_sql_succeeds():
    task = _task(
        '{"items":[{"name":"result","value":8.10725}]}',
        sql_call_records=[
            {"table_name": "other_table", "success": True},
            {"table_name": "other_table", "success": True},
        ],
    )
    wrong = eco_qa_reward_function(task, 'FINAL ANSWER: {"items":[{"name":"result","value":7.0}]}')
    assert wrong.reward == 0.0
    assert wrong.metadata["exp_table_hit_rate"] == 0.0
    assert wrong.metadata["exp_table_sql_succ_rate"] == 0.0


def test_incorrect_answer_csv_suffix_table_still_counts_as_expected_hit():
    task = _task(
        '{"items":[{"name":"result","value":8.10725}]}',
        sql_call_records=[
            {"table_name": "interest_rates.csv", "success": True},
            {"table_name": "interest_rates.csv", "success": True},
        ],
    )
    wrong = eco_qa_reward_function(task, 'FINAL ANSWER: {"items":[{"name":"result","value":7.0}]}')
    assert wrong.reward > 0.0 and wrong.reward < 1.0
    assert wrong.metadata["exp_table_hit_rate"] == 1.0
    assert wrong.metadata["exp_table_sql_succ_rate"] == 1.0


def test_incorrect_non_structure_prediction_gets_no_shaping_bonus():
    task = _task(
        '{"items":[{"name":"result","value":8.10725}]}',
        sql_call_records=[
            {"table_name": "interest_rates", "success": True},
            {"table_name": "interest_rates", "success": True},
        ],
    )
    wrong = eco_qa_reward_function(task, "FINAL ANSWER: 8.0")
    assert wrong.reward == 0.0
    assert wrong.metadata["pred_structure_valid"] == 0.0
