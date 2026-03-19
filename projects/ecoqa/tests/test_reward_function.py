from projects.ecoqa.eco_qa_reward import eco_qa_reward_function


def _task(
    ground_truth: str,
    *,
    sql_call_records: list[dict] | None = None,
):
    return {
        "question": "dummy",
        "ground_truth": ground_truth,
        "table_name": "interest_rates",
        "sql_call_records": sql_call_records if sql_call_records is not None else [],
    }


def test_structure_numeric_value_match():
    task = _task('{"rows":[{"result":8.10725}]}')
    correct = eco_qa_reward_function(task, 'FINAL ANSWER: {"rows":[{"result":8.10725}]}')
    wrong = eco_qa_reward_function(task, 'FINAL ANSWER: {"rows":[{"result":7.0}]}')
    assert correct.reward == 1.0 and correct.is_correct
    assert wrong.reward == 0.0 and not wrong.is_correct


def test_structure_numeric_value_allows_percent_symbol():
    task = _task('{"rows":[{"rate":5.5}]}')
    pred = eco_qa_reward_function(task, 'FINAL ANSWER: {"rows":[{"rate":"5.5%"}]}')
    assert pred.reward == 1.0 and pred.is_correct


def test_structure_accepts_json_code_block_without_final_answer_prefix():
    task = _task('{"rows":[{"rate":5.5}]}')
    pred = eco_qa_reward_function(task, '```json\n{"rows":[{"rate":"5.5%"}]}\n```')
    assert pred.reward == 1.0 and pred.is_correct


def test_structure_text_value_match():
    task = _task('{"rows":[{"geo_name":"广东"}]}')
    pred = eco_qa_reward_function(task, 'FINAL ANSWER: {"rows":[{"geo_name":"广东"}]}')
    wrong = eco_qa_reward_function(task, 'FINAL ANSWER: {"rows":[{"geo_name":"江苏"}]}')
    assert pred.reward == 1.0 and pred.is_correct
    assert wrong.reward == 0.0 and not wrong.is_correct


def test_structure_multiple_rows_match_order_insensitive():
    gt = '{"rows":[{"year":2024,"month":1,"a":1},{"b":"2"}]}'
    task = _task(gt)
    pred = eco_qa_reward_function(
        task,
        'FINAL ANSWER: {"rows":[{"b":2.0},{"month":1,"a":"1","year":"2024"}]}',
    )
    assert pred.reward == 1.0 and pred.is_correct


def test_structure_requires_same_keys_and_values():
    task = _task('{"rows":[{"year":2024,"month":9}]}')
    wrong = eco_qa_reward_function(task, 'FINAL ANSWER: {"rows":[{"month":9}]}')
    assert wrong.reward == 0.0 and not wrong.is_correct


def test_structure_no_partial_reward_for_wrong_schema():
    gt = '{"rows":[{"geo_name":"广东"},{"m2_100m_cny":411709.31}]}'
    task = _task(gt)
    wrong = eco_qa_reward_function(task, 'FINAL ANSWER: {"items":[{"name":"geo_name","value":"广东"}]}')
    assert wrong.reward == 0.0 and not wrong.is_correct


def test_structure_empty_rows_match():
    task = _task('{"rows":[]}')
    pred = eco_qa_reward_function(task, 'FINAL ANSWER: {"rows":[]}')
    wrong = eco_qa_reward_function(task, 'FINAL ANSWER: {"rows":[{"x":1}]}')
    assert pred.reward == 1.0 and pred.is_correct
    assert wrong.reward == 0.0 and not wrong.is_correct


def test_reward_metadata_only_contains_required_keys():
    task = _task('{"rows":[]}')
    pred = eco_qa_reward_function(task, 'FINAL ANSWER: {"rows":[]}')
    assert set(pred.metadata.keys()) == {
        "final_reward",
        "correctness_reward",
        "shaping_bonus",
        "exp_table_hit_rate",
        "exp_table_sql_succ_rate",
    }


def test_incorrect_answer_gets_progress_bonus_when_right_table_and_sql_success():
    task = _task(
        '{"rows":[{"result":8.10725}]}',
        sql_call_records=[
            {"table_name": "interest_rates", "success": True},
            {"table_name": "interest_rates", "success": True},
        ],
    )
    wrong = eco_qa_reward_function(task, 'FINAL ANSWER: {"rows":[{"result":7.0}]}')
    assert wrong.reward > 0.0 and wrong.reward < 1.0
    assert wrong.metadata["exp_table_hit_rate"] == 1.0
    assert wrong.metadata["exp_table_sql_succ_rate"] == 1.0


def test_incorrect_answer_gets_no_bonus_without_sql_success():
    task = _task(
        '{"rows":[{"result":8.10725}]}',
        sql_call_records=[
            {"table_name": "interest_rates", "success": False},
            {"table_name": "interest_rates", "success": False},
        ],
    )
    wrong = eco_qa_reward_function(task, 'FINAL ANSWER: {"rows":[{"result":7.0}]}')
    assert wrong.reward == 0.0
    assert wrong.metadata["exp_table_hit_rate"] == 1.0
    assert wrong.metadata["exp_table_sql_succ_rate"] == 0.0


def test_incorrect_answer_gets_no_bonus_when_only_wrong_table_sql_succeeds():
    task = _task(
        '{"rows":[{"result":8.10725}]}',
        sql_call_records=[
            {"table_name": "other_table", "success": True},
            {"table_name": "other_table", "success": True},
        ],
    )
    wrong = eco_qa_reward_function(task, 'FINAL ANSWER: {"rows":[{"result":7.0}]}')
    assert wrong.reward == 0.0
    assert wrong.metadata["exp_table_hit_rate"] == 0.0
    assert wrong.metadata["exp_table_sql_succ_rate"] == 0.0


def test_incorrect_answer_csv_suffix_table_still_counts_as_expected_hit():
    task = _task(
        '{"rows":[{"result":8.10725}]}',
        sql_call_records=[
            {"table_name": "interest_rates.csv", "success": True},
            {"table_name": "interest_rates.csv", "success": True},
        ],
    )
    wrong = eco_qa_reward_function(task, 'FINAL ANSWER: {"rows":[{"result":7.0}]}')
    assert wrong.reward > 0.0 and wrong.reward < 1.0
    assert wrong.metadata["exp_table_hit_rate"] == 1.0
    assert wrong.metadata["exp_table_sql_succ_rate"] == 1.0


def test_incorrect_answer_json_code_block_without_final_answer_still_gets_shaping_bonus():
    task = _task(
        '{"rows":[{"result":8.10725}]}',
        sql_call_records=[
            {"table_name": "interest_rates", "success": True},
            {"table_name": "interest_rates", "success": True},
        ],
    )
    wrong = eco_qa_reward_function(task, '```json\n{"rows":[{"result":7.0}]}\n```')
    assert wrong.reward > 0.0 and wrong.reward < 1.0


def test_incorrect_non_structure_prediction_gets_no_shaping_bonus():
    task = _task(
        '{"rows":[{"result":8.10725}]}',
        sql_call_records=[
            {"table_name": "interest_rates", "success": True},
            {"table_name": "interest_rates", "success": True},
        ],
    )
    wrong = eco_qa_reward_function(task, "FINAL ANSWER: 8.0")
    assert wrong.reward == 0.0


def test_incorrect_partial_field_match_gets_more_bonus_than_full_miss():
    task = _task(
        '{"rows":[{"year":2024,"result":8.10725}]}',
        sql_call_records=[
            {"table_name": "interest_rates", "success": True},
            {"table_name": "interest_rates", "success": True},
        ],
    )
    partial = eco_qa_reward_function(task, 'FINAL ANSWER: {"rows":[{"year":2024,"result":7.0}]}')
    full_miss = eco_qa_reward_function(task, 'FINAL ANSWER: {"rows":[{"year":2023,"result":7.0}]}')

    assert partial.reward > 0.0
    assert full_miss.reward == 0.0
    assert partial.reward > full_miss.reward
