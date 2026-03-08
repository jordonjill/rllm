import re

from rllm.rewards.reward_types import RewardOutput

_FINAL_ANSWER_CODE_BLOCK_RE = re.compile(r"```\s*FINAL ANSWER:\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_FINAL_ANSWER_PARAGRAPH_RE = re.compile(r"FINAL ANSWER:\s*(.*?)(?=\n\s*\n)", re.DOTALL | re.IGNORECASE)
_FINAL_ANSWER_TAIL_RE = re.compile(r"FINAL ANSWER:\s*(.*)$", re.DOTALL | re.IGNORECASE)
_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?%?")


def _extract_final_answer(action: str) -> str:
    match = _FINAL_ANSWER_CODE_BLOCK_RE.search(action)
    if match:
        return match.group(1).strip()

    match = _FINAL_ANSWER_PARAGRAPH_RE.search(action)
    if match:
        return match.group(1).strip()

    match = _FINAL_ANSWER_TAIL_RE.search(action)
    if match:
        return match.group(1).strip()

    return action.strip()


def _parse_number(text: str) -> tuple[float | None, bool]:
    if not text:
        return None, False

    cleaned = text.strip().replace("\\boxed{", "").replace("}", "")
    match = _NUMBER_RE.search(cleaned)
    if not match:
        return None, False

    token = match.group(0)
    is_percent = token.endswith("%")
    token = token.rstrip("%")
    token = token.replace(",", "")

    try:
        value = float(token)
    except ValueError:
        return None, is_percent

    return value, is_percent


def _normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _check_right_table_accessed(accessed_tables: list[str], expected_table_name: str | list[str]) -> float:
    if not accessed_tables or not expected_table_name:
        return 0.0

    normalized_access = {table.strip().lower() for table in accessed_tables if isinstance(table, str) and table.strip()}

    if isinstance(expected_table_name, list):
        expected = [name.strip().lower() for name in expected_table_name if isinstance(name, str) and name.strip()]
    else:
        expected = [expected_table_name.strip().lower()] if isinstance(expected_table_name, str) and expected_table_name.strip() else []

    if not expected:
        return 0.0

    hits = sum(1 for name in expected if name in normalized_access)
    return hits / len(expected)


def eco_qa_reward_function(task_info: dict, action: str) -> RewardOutput:
    question = task_info.get("question")
    ground_truth = task_info.get("ground_truth")

    if not action or not question or ground_truth in (None, ""):
        return RewardOutput(reward=0.0, is_correct=False, metadata={"correctness_reward": 0.0, "right_table_access_reward": 0.0})

    final_answer = _extract_final_answer(action)

    pred_num, pred_pct = _parse_number(final_answer)
    gt_num, gt_pct = _parse_number(str(ground_truth))

    is_correct = False

    if pred_num is not None and gt_num is not None:
        # Allow decimal vs percentage equivalence.
        if pred_pct and not gt_pct:
            pred_num = pred_num / 100.0
        if gt_pct and not pred_pct:
            gt_num = gt_num / 100.0

        tol = max(1e-4, 1e-3 * max(abs(gt_num), 1.0))
        is_correct = abs(pred_num - gt_num) <= tol
    else:
        is_correct = _normalize_text(final_answer) == _normalize_text(str(ground_truth))

    correctness_reward = 1.0 if is_correct else 0.0

    accessed_tables = task_info.get("accessed_tables", [])
    expected_table_name = task_info.get("table_name", "")
    right_table_access_reward = _check_right_table_accessed(accessed_tables, expected_table_name)

    return RewardOutput(
        reward=correctness_reward,
        is_correct=is_correct,
        metadata={
            "correctness_reward": correctness_reward,
            "right_table_access_reward": right_table_access_reward,
            "final_answer_extracted": final_answer,
        },
    )
