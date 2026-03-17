import json
import math
import os
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
    "no_data",
    "No Data",
    "null",
    "超出数据范围",
    "数据不存在",
    "no data",
    "not found",
    "no matching data",
    "empty result",
)
_TEMPORAL_KEY_HINTS = ("date", "year", "month", "day", "quarter", "qtr", "period", "ref_date", "week")
_DATE_YMD_RE = re.compile(r"^\s*(\d{4})[-/年](\d{1,2})(?:[-/月](\d{1,2}))?(?:日)?\s*$")
_YEAR_QUARTER_RE = re.compile(r"^\s*(\d{4})\s*[-_/ ]?q([1-4])\s*$", re.IGNORECASE)
_YEAR_ONLY_RE = re.compile(r"^\s*(\d{4})\s*$")
_MONTH_CN_RE = re.compile(r"^\s*(1[0-2]|0?[1-9])月\s*$")

def _as_env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _as_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    return parsed


_SHAPING_ENABLED = _as_env_bool("ECOQA_ENABLE_SHAPING_BONUS", True)
_MAX_SHAPING_BONUS = max(0.0, _as_env_float("ECOQA_MAX_SHAPING_BONUS", 0.15))


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


def _is_no_data_prediction(text: str) -> bool:
    if not isinstance(text, str) or not text.strip():
        return False

    parsed = _try_parse_json(text)
    if isinstance(parsed, dict):
        pred_type = _normalize_text(str(parsed.get("type", "")))
        if pred_type in {"no_data", "nodata"}:
            return True

        for key in ("reason", "message", "value", "answer"):
            value = parsed.get(key)
            if isinstance(value, str) and _is_no_data_prediction(value):
                return True

    normalized = _normalize_text(text)
    if "无数据" in normalized:
        return True
    return bool(re.search(r"\bno[\s_]*data\b", normalized))


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


def _serialize_value(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _serialize_row_values_unordered(row: dict) -> str:
    normalized_row = _normalize_row(row)
    value_tokens = sorted(_serialize_value(value) for value in normalized_row.values())
    return json.dumps(value_tokens, ensure_ascii=False, separators=(",", ":"))


def _list_alias_value_match(pred_rows: list, gt_rows: list) -> bool:
    if not pred_rows or not gt_rows:
        return False

    if len(pred_rows) != len(gt_rows):
        return False

    if not all(isinstance(row, dict) for row in pred_rows):
        return False
    if not all(isinstance(row, dict) for row in gt_rows):
        return False

    pred_counter = Counter(_serialize_row_values_unordered(row) for row in pred_rows)
    gt_counter = Counter(_serialize_row_values_unordered(row) for row in gt_rows)
    return pred_counter == gt_counter


def _is_temporal_key(key: str) -> bool:
    key_norm = str(key).strip().lower()
    return any(token in key_norm for token in _TEMPORAL_KEY_HINTS)


def _parse_temporal_parts_from_string(value: str) -> dict[str, int]:
    text = value.strip()
    if not text:
        return {}

    match = _DATE_YMD_RE.match(text)
    if match:
        parts = {"year": int(match.group(1)), "month": int(match.group(2))}
        day = match.group(3)
        if day is not None:
            parts["day"] = int(day)
        return parts

    match = _YEAR_QUARTER_RE.match(text)
    if match:
        return {"year": int(match.group(1)), "quarter": int(match.group(2))}

    match = _YEAR_ONLY_RE.match(text)
    if match:
        return {"year": int(match.group(1))}

    match = _MONTH_CN_RE.match(text)
    if match:
        return {"month": int(match.group(1))}

    return {}


def _extract_temporal_parts_for_cell(key: str, value) -> dict[str, int]:
    key_norm = str(key).strip().lower()
    parts: dict[str, int] = {}

    if isinstance(value, int | float) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number):
            ivalue = int(round(number))
            if "year" in key_norm and 1900 <= ivalue <= 2100:
                parts["year"] = ivalue
            if "month" in key_norm and 1 <= ivalue <= 12:
                parts["month"] = ivalue
            if "day" in key_norm and 1 <= ivalue <= 31:
                parts["day"] = ivalue
            if ("quarter" in key_norm or "qtr" in key_norm) and 1 <= ivalue <= 4:
                parts["quarter"] = ivalue

    if isinstance(value, str):
        parsed = _parse_temporal_parts_from_string(value)
        # Parse date-like strings as temporal by default, and also trust explicit temporal keys.
        if parsed and (_is_temporal_key(key_norm) or any(ch in value for ch in ("-", "/", "年", "月", "Q", "q"))):
            parts.update(parsed)

    return parts


def _split_row_non_temporal_and_temporal(row: dict) -> tuple[Counter, dict[str, int]]:
    non_temporal_tokens: list[str] = []
    temporal_parts: dict[str, int] = {}

    for key, value in row.items():
        parts = _extract_temporal_parts_for_cell(str(key), value)
        if parts:
            temporal_parts.update(parts)
        else:
            non_temporal_tokens.append(_serialize_value(_normalize_json_value(value)))

    return Counter(non_temporal_tokens), temporal_parts


def _rows_temporal_compatible(pred_row: dict, gt_row: dict) -> bool:
    pred_non_temporal, pred_temporal = _split_row_non_temporal_and_temporal(pred_row)
    gt_non_temporal, gt_temporal = _split_row_non_temporal_and_temporal(gt_row)

    if pred_non_temporal != gt_non_temporal:
        return False

    if not pred_temporal and not gt_temporal:
        return True

    shared_keys = set(pred_temporal).intersection(gt_temporal)
    if not shared_keys:
        return False

    return all(pred_temporal[key] == gt_temporal[key] for key in shared_keys)


def _list_temporal_value_match(pred_rows: list, gt_rows: list) -> bool:
    if not pred_rows or not gt_rows or len(pred_rows) != len(gt_rows):
        return False
    if not all(isinstance(row, dict) for row in pred_rows):
        return False
    if not all(isinstance(row, dict) for row in gt_rows):
        return False

    # Bipartite matching between predicted rows and GT rows under temporal compatibility.
    adjacency: list[list[int]] = []
    for pred_row in pred_rows:
        candidates = [idx for idx, gt_row in enumerate(gt_rows) if _rows_temporal_compatible(pred_row, gt_row)]
        adjacency.append(candidates)

    if any(len(candidates) == 0 for candidates in adjacency):
        return False

    matched_gt_for_pred = [-1] * len(pred_rows)
    matched_pred_for_gt = [-1] * len(gt_rows)

    def _dfs(pred_idx: int, seen_gt: set[int]) -> bool:
        for gt_idx in adjacency[pred_idx]:
            if gt_idx in seen_gt:
                continue
            seen_gt.add(gt_idx)
            current_pred = matched_pred_for_gt[gt_idx]
            if current_pred == -1 or _dfs(current_pred, seen_gt):
                matched_gt_for_pred[pred_idx] = gt_idx
                matched_pred_for_gt[gt_idx] = pred_idx
                return True
        return False

    for pred_idx in range(len(pred_rows)):
        if not _dfs(pred_idx, set()):
            return False
    return True


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
        return "no_data"
    if answer_type in {"list", "scalar"}:
        return answer_type

    parsed_gt = _try_parse_json(ground_truth)
    if isinstance(parsed_gt, list):
        return "list"
    gt_num, _ = _parse_number(ground_truth)
    if gt_num is not None:
        return "scalar"
    if _is_no_data_text(ground_truth):
        return "no_data"
    return "text"


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _normalize_expected_tables(expected_table_name: str | list[str]) -> set[str]:
    if isinstance(expected_table_name, list):
        expected = [name.strip().lower() for name in expected_table_name if isinstance(name, str) and name.strip()]
    elif isinstance(expected_table_name, str) and expected_table_name.strip():
        expected = [expected_table_name.strip().lower()]
    else:
        expected = []
    return set(expected)


def _sql_call_stats(task_info: dict, expected_tables: set[str]) -> tuple[int, int, int]:
    records = task_info.get("sql_call_records", [])
    if not isinstance(records, list):
        return 0, 0, 0

    total_sql_calls = 0
    exp_table_sql_calls = 0
    exp_table_sql_success = 0
    for record in records:
        if not isinstance(record, dict):
            continue
        total_sql_calls += 1
        table_name = str(record.get("table_name", "")).strip().lower()
        if table_name not in expected_tables:
            continue
        exp_table_sql_calls += 1
        if bool(record.get("success", False)):
            exp_table_sql_success += 1
    return total_sql_calls, exp_table_sql_calls, exp_table_sql_success


def eco_qa_reward_function(task_info: dict, action: str) -> RewardOutput:
    question = task_info.get("question")
    ground_truth = task_info.get("ground_truth")
    if not action or not question or ground_truth in (None, ""):
        return RewardOutput(reward=0.0, is_correct=False, metadata={"correctness_reward": 0.0})

    ground_truth_text = str(ground_truth)
    final_answer = _extract_final_answer(action)
    target_kind = _determine_target_kind(task_info, ground_truth_text)

    is_correct = False
    list_exact_match = False
    list_alias_value_match = False
    list_temporal_value_match = False

    if target_kind == "scalar":
        pred_num, _ = _extract_pred_scalar(final_answer)
        gt_num, _ = _parse_number(ground_truth_text)
        if pred_num is not None and gt_num is not None:
            tol = max(1e-4, 1e-3 * max(abs(gt_num), 1.0))
            is_correct = abs(pred_num - gt_num) <= tol
    elif target_kind == "list":
        pred_rows = _extract_pred_rows(final_answer)
        gt_rows = _try_parse_json(ground_truth_text)
        if isinstance(pred_rows, list) and isinstance(gt_rows, list):
            pred_dict_rows = [row for row in pred_rows if isinstance(row, dict)]
            gt_dict_rows = [row for row in gt_rows if isinstance(row, dict)]

            if len(pred_dict_rows) == len(pred_rows) and len(gt_dict_rows) == len(gt_rows):
                pred_counter = Counter(_serialize_row(_normalize_row(row)) for row in pred_dict_rows)
                gt_counter = Counter(_serialize_row(_normalize_row(row)) for row in gt_dict_rows)
                list_exact_match = pred_counter == gt_counter and len(gt_counter) > 0

            list_alias_value_match = _list_alias_value_match(pred_rows, gt_rows)
            list_temporal_value_match = _list_temporal_value_match(pred_rows, gt_rows)
            is_correct = list_exact_match or list_alias_value_match or list_temporal_value_match
    elif target_kind == "no_data":
        is_correct = _is_no_data_prediction(final_answer)
    else:
        pred_text = _extract_pred_text(final_answer)
        is_correct = _normalize_text(pred_text) == _normalize_text(ground_truth_text)

    correctness_reward = 1.0 if is_correct else 0.0
    expected_table_name = task_info.get("table_name", "")

    expected_tables = _normalize_expected_tables(expected_table_name)
    sql_call_count, exp_table_sql_calls, exp_table_sql_success = _sql_call_stats(task_info, expected_tables)
    exp_table_hit_rate = _safe_ratio(exp_table_sql_calls, sql_call_count)
    exp_table_sql_succ_rate = _safe_ratio(exp_table_sql_success, exp_table_sql_calls)

    # Shaping reward for incorrect answers is based only on SQL success ratio
    # over expected table calls.
    shaping_bonus = 0.0
    if _SHAPING_ENABLED and not is_correct and exp_table_sql_calls > 0:
        shaping_bonus = max(0.0, min(_MAX_SHAPING_BONUS, _MAX_SHAPING_BONUS * exp_table_sql_succ_rate))

    final_reward = correctness_reward if is_correct else shaping_bonus

    return RewardOutput(
        reward=final_reward,
        is_correct=is_correct,
        metadata={
            "correctness_reward": correctness_reward,
            "shaping_bonus": shaping_bonus,
            "final_reward": final_reward,
            "sql_call_count": sql_call_count,
            "exp_table_sql_calls": exp_table_sql_calls,
            "exp_table_sql_success": exp_table_sql_success,
            "exp_table_hit_rate": exp_table_hit_rate,
            "exp_table_sql_succ_rate": exp_table_sql_succ_rate,
            "shaping_enabled": float(_SHAPING_ENABLED),
            "max_shaping_bonus": _MAX_SHAPING_BONUS,
            "target_kind": target_kind,
            "list_exact_match": list_exact_match,
            "list_alias_value_match": list_alias_value_match,
            "list_temporal_value_match": list_temporal_value_match,
            "final_answer_extracted": final_answer,
        },
    )
