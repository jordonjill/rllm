import json
import math
import re
from collections import Counter

from rllm.rewards.reward_types import RewardOutput

_FINAL_ANSWER_CODE_BLOCK_RE = re.compile(r"```\s*FINAL ANSWER:\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_FINAL_ANSWER_PARAGRAPH_RE = re.compile(r"FINAL ANSWER:\s*(.*?)(?=\n\s*\n)", re.DOTALL | re.IGNORECASE)
_FINAL_ANSWER_TAIL_RE = re.compile(r"FINAL ANSWER:\s*(.*)$", re.DOTALL | re.IGNORECASE)
_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?%?")
_FULL_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?%?")
_NO_DATA_KEYWORDS = (
    "无数据",
    "查不到",
    "无法查询",
    "数据范围仅覆盖",
    "超出数据范围",
    "数据不存在",
    "no data",
    "not found",
    "no matching data",
    "empty result",
)
_ERROR_CLASS_KEYWORDS = {
    "out_of_range": ("数据范围仅覆盖", "数据起始年份", "无法查询", "年份不存在", "数据不存在", "out of range", "outside"),
    "logical_conflict": ("逻辑冲突", "不能同时", "不可能", "conflict"),
    "unrealistic_condition": ("合理范围", "不存在超过", "异常记录不存在", "unrealistic"),
    "no_data": _NO_DATA_KEYWORDS,
}


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


def _try_parse_json(text: str):
    if not text or not isinstance(text, str):
        return None

    stripped = text.strip()
    if not stripped:
        return None

    try:
        return json.loads(stripped)
    except Exception:
        pass

    for left, right in (("{", "}"), ("[", "]")):
        start = stripped.find(left)
        end = stripped.rfind(right)
        if start == -1 or end <= start:
            continue
        candidate = stripped[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            continue

    return None


def _parse_number(text: str) -> tuple[float | None, bool]:
    if not text:
        return None, False

    cleaned = text.strip().replace("\\boxed{", "").replace("}", "")
    match = _NUMBER_RE.search(cleaned)
    if not match:
        return None, False

    token = match.group(0)
    is_percent = token.endswith("%")
    token = token.rstrip("%").replace(",", "")

    try:
        value = float(token)
    except ValueError:
        return None, is_percent

    return value, is_percent


def _normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _is_no_data_text(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False
    return any(keyword in normalized for keyword in _NO_DATA_KEYWORDS)


def _classify_error_text(text: str) -> str:
    normalized = _normalize_text(text)
    for klass, keywords in _ERROR_CLASS_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            return klass
    return ""


def _coerce_numeric_string(value: str) -> float | None:
    token = value.strip().replace(" ", "")
    if not token or not _FULL_NUMBER_RE.fullmatch(token):
        return None

    is_percent = token.endswith("%")
    token = token.rstrip("%").replace(",", "")
    try:
        parsed = float(token)
    except ValueError:
        return None
    if is_percent:
        parsed /= 100.0
    return parsed


def _normalize_json_value(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        numeric = float(value)
        if math.isfinite(numeric):
            return round(numeric, 6)
        return str(value)
    if isinstance(value, str):
        numeric = _coerce_numeric_string(value)
        if numeric is not None:
            return round(numeric, 6)
        return _normalize_text(value)
    if isinstance(value, list):
        return [_normalize_json_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k).strip().lower(): _normalize_json_value(v) for k, v in value.items()}
    return _normalize_text(str(value))


def _normalize_row(row: dict) -> dict:
    return {str(key).strip().lower(): _normalize_json_value(value) for key, value in row.items()}


def _serialize_row(row: dict) -> str:
    return json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _extract_pred_rows(final_answer: str):
    parsed = _try_parse_json(final_answer)
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        for key in ("rows", "value", "data", "answer"):
            candidate = parsed.get(key)
            if isinstance(candidate, list):
                return candidate
    return None


def _extract_pred_scalar(final_answer: str) -> tuple[float | None, bool]:
    parsed = _try_parse_json(final_answer)
    if isinstance(parsed, dict):
        value = parsed.get("value")
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value), False
        if isinstance(value, str):
            return _parse_number(value)
    elif isinstance(parsed, int | float) and not isinstance(parsed, bool):
        return float(parsed), False
    elif isinstance(parsed, str):
        return _parse_number(parsed)
    return _parse_number(final_answer)


def _extract_pred_text(final_answer: str) -> str:
    parsed = _try_parse_json(final_answer)
    if isinstance(parsed, dict):
        for key in ("reason", "message", "value", "answer"):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                return value
    if isinstance(parsed, str):
        return parsed
    return final_answer


def _determine_target_kind(task_info: dict, ground_truth: str) -> str:
    question_type = str(task_info.get("question_type", "")).strip().lower()
    answer_type = str(task_info.get("answer_type", "")).strip().lower()

    if question_type == "single_table_error":
        return "error"
    if answer_type in {"list", "scalar"}:
        return answer_type

    parsed_gt = _try_parse_json(ground_truth)
    if isinstance(parsed_gt, list):
        return "list"
    gt_num, _ = _parse_number(ground_truth)
    if gt_num is not None:
        return "scalar"
    if _is_no_data_text(ground_truth):
        return "error"
    return "text"


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

    ground_truth_text = str(ground_truth)
    final_answer = _extract_final_answer(action)
    target_kind = _determine_target_kind(task_info, ground_truth_text)

    is_correct = False
    list_exact_match = False

    if target_kind == "scalar":
        pred_num, pred_pct = _extract_pred_scalar(final_answer)
        gt_num, gt_pct = _parse_number(ground_truth_text)
        if pred_num is not None and gt_num is not None:
            if pred_pct and not gt_pct:
                pred_num /= 100.0
            if gt_pct and not pred_pct:
                gt_num /= 100.0
            tol = max(1e-4, 1e-3 * max(abs(gt_num), 1.0))
            is_correct = abs(pred_num - gt_num) <= tol
    elif target_kind == "list":
        pred_rows = _extract_pred_rows(final_answer)
        gt_rows = _try_parse_json(ground_truth_text)
        if isinstance(pred_rows, list) and isinstance(gt_rows, list):
            pred_counter = Counter(_serialize_row(_normalize_row(row)) for row in pred_rows if isinstance(row, dict))
            gt_counter = Counter(_serialize_row(_normalize_row(row)) for row in gt_rows if isinstance(row, dict))
            list_exact_match = pred_counter == gt_counter and len(gt_counter) > 0
            is_correct = list_exact_match
    elif target_kind == "error":
        pred_text = _extract_pred_text(final_answer)
        if _normalize_text(pred_text) == _normalize_text(ground_truth_text):
            is_correct = True
        else:
            gt_class = _classify_error_text(ground_truth_text)
            pred_class = _classify_error_text(pred_text)
            if gt_class and gt_class == pred_class:
                is_correct = True
            elif _is_no_data_text(pred_text) and _is_no_data_text(ground_truth_text):
                is_correct = True
    else:
        pred_text = _extract_pred_text(final_answer)
        is_correct = _normalize_text(pred_text) == _normalize_text(ground_truth_text)

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
            "target_kind": target_kind,
            "list_exact_match": list_exact_match,
            "final_answer_extracted": final_answer,
        },
    )
