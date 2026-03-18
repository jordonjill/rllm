import json
import math
import os
import re
from collections import Counter

from rllm.rewards.reward_types import RewardOutput

_FINAL_ANSWER_CODE_BLOCK_RE = re.compile(r"```(?:\w+)?\s*FINAL ANSWER:\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_JSON_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_FINAL_ANSWER_PARAGRAPH_RE = re.compile(r"FINAL ANSWER:\s*(.*?)(?=\n\s*\n)", re.DOTALL | re.IGNORECASE)
_FINAL_ANSWER_TAIL_RE = re.compile(r"FINAL ANSWER:\s*(.*)$", re.DOTALL | re.IGNORECASE)
_FULL_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?%?")
_QUALIFIER_KEYS = {
    "year",
    "quarter",
    "month",
    "week",
    "day",
    "ref_date",
    "date",
    "geo_level",
    "geo_code",
    "geo_name",
    "parent_geo_code",
    "parent_geo_name",
    "sector",
    "industry",
    "category",
    "rank",
}


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
_MAX_SHAPING_BONUS = max(0.0, _as_env_float("ECOQA_MAX_SHAPING_BONUS", 0.10))


def _extract_final_answer_with_status(action: str) -> tuple[str, bool]:
    match = _FINAL_ANSWER_CODE_BLOCK_RE.search(action)
    if match:
        return match.group(1).strip(), True

    for code_block in _JSON_CODE_BLOCK_RE.finditer(action):
        candidate = code_block.group(1).strip()
        if _try_parse_json(candidate) is not None:
            return candidate, True

    match = _FINAL_ANSWER_PARAGRAPH_RE.search(action)
    if match:
        return match.group(1).strip(), True

    match = _FINAL_ANSWER_TAIL_RE.search(action)
    if match:
        return match.group(1).strip(), True

    stripped = action.strip()
    if _try_parse_json(stripped) is not None:
        return stripped, True
    return stripped, False


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


def _normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _coerce_numeric_string(value: str) -> float | None:
    token = value.strip().replace(" ", "")
    if not token or not _FULL_NUMBER_RE.fullmatch(token):
        return None

    token = token.rstrip("%").replace(",", "")
    try:
        parsed = float(token)
    except ValueError:
        return None
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
    normalized = {str(key).strip().lower(): _normalize_json_value(value) for key, value in row.items()}
    if "result" in normalized:
        return normalized

    keys = list(normalized.keys())
    if not keys:
        return normalized

    non_qualifier_keys = [key for key in keys if key not in _QUALIFIER_KEYS]

    # Alias-tolerant normalization: if the row has a single metric-like key,
    # canonicalize it to "result" so SQL alias differences do not hurt reward.
    if len(non_qualifier_keys) == 1:
        key = non_qualifier_keys[0]
        normalized["result"] = normalized.pop(key)
        return normalized

    # For one-column target rows (e.g., month/geo_name lists), allow the
    # single field to act as result as well.
    if len(keys) == 1:
        only_key = keys[0]
        normalized = {"result": normalized[only_key]}

    return normalized


def _serialize_row(row: dict) -> str:
    return json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _extract_rows(text: str) -> list | None:
    parsed = _try_parse_json(text)
    if not isinstance(parsed, dict):
        return None
    rows = parsed.get("rows")
    if isinstance(rows, list):
        return rows
    return None


def _is_valid_structure_prediction(text: str) -> bool:
    parsed = _try_parse_json(text)
    if not isinstance(parsed, dict):
        return False

    rows = parsed.get("rows")
    if not isinstance(rows, list):
        return False
    return all(isinstance(row, dict) for row in rows)


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _normalize_table_token(table_name: object) -> str:
    if not isinstance(table_name, str):
        return ""
    token = table_name.strip().lower()
    if token.endswith(".csv"):
        token = token[:-4]
    return token


def _normalize_expected_tables(expected_table_name: str | list[str]) -> set[str]:
    if isinstance(expected_table_name, list):
        expected = [_normalize_table_token(name) for name in expected_table_name if isinstance(name, str) and name.strip()]
    elif isinstance(expected_table_name, str) and expected_table_name.strip():
        expected = [_normalize_table_token(expected_table_name)]
    else:
        expected = []
    return {name for name in expected if name}


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
        table_name = _normalize_table_token(record.get("table_name", ""))
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
        return RewardOutput(
            reward=0.0,
            is_correct=False,
            metadata={
                "final_reward": 0.0,
                "correctness_reward": 0.0,
                "shaping_bonus": 0.0,
                "exp_table_hit_rate": 0.0,
                "exp_table_sql_succ_rate": 0.0,
            },
        )

    ground_truth_text = str(ground_truth)
    final_answer, final_answer_extractable = _extract_final_answer_with_status(action)

    is_correct = False
    gt_rows = _extract_rows(ground_truth_text)
    pred_rows = _extract_rows(final_answer)
    pred_structure_valid = _is_valid_structure_prediction(final_answer)

    if isinstance(gt_rows, list) and isinstance(pred_rows, list):
        if all(isinstance(row, dict) for row in gt_rows) and all(isinstance(row, dict) for row in pred_rows):
            gt_counter = Counter(_serialize_row(_normalize_row(row)) for row in gt_rows)
            pred_counter = Counter(_serialize_row(_normalize_row(row)) for row in pred_rows)
            is_correct = pred_counter == gt_counter

    correctness_reward = 1.0 if is_correct else 0.0
    expected_table_name = task_info.get("table_name", "")

    expected_tables = _normalize_expected_tables(expected_table_name)
    sql_call_count, exp_table_sql_calls, exp_table_sql_success = _sql_call_stats(task_info, expected_tables)
    exp_table_hit_rate = _safe_ratio(exp_table_sql_calls, sql_call_count)
    exp_table_sql_succ_rate = _safe_ratio(exp_table_sql_success, exp_table_sql_calls)

    shaping_bonus = 0.0
    answer_score = 1.0 if final_answer_extractable and pred_structure_valid else 0.0
    if _SHAPING_ENABLED and not is_correct and sql_call_count > 0 and exp_table_sql_calls > 0:
        shaping_factor = exp_table_hit_rate * exp_table_sql_succ_rate * answer_score
        shaping_bonus = max(0.0, min(_MAX_SHAPING_BONUS, _MAX_SHAPING_BONUS * shaping_factor))

    final_reward = correctness_reward if is_correct else shaping_bonus

    return RewardOutput(
        reward=final_reward,
        is_correct=is_correct,
        metadata={
            "final_reward": final_reward,
            "correctness_reward": correctness_reward,
            "shaping_bonus": shaping_bonus,
            "exp_table_hit_rate": exp_table_hit_rate,
            "exp_table_sql_succ_rate": exp_table_sql_succ_rate,
        },
    )
