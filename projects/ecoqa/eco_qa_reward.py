import json
import math
import os
import re
from collections import Counter

from rllm.rewards.reward_types import RewardOutput

_FINAL_ANSWER_CODE_BLOCK_RE = re.compile(r"```\s*FINAL ANSWER:\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_FINAL_ANSWER_PARAGRAPH_RE = re.compile(r"FINAL ANSWER:\s*(.*?)(?=\n\s*\n)", re.DOTALL | re.IGNORECASE)
_FINAL_ANSWER_TAIL_RE = re.compile(r"FINAL ANSWER:\s*(.*)$", re.DOTALL | re.IGNORECASE)
_FULL_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?%?")


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
_MAX_SHAPING_BONUS = max(0.0, _as_env_float("ECOQA_MAX_SHAPING_BONUS", 0.05))


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


def _normalize_item(item: dict) -> dict:
    return {str(key).strip().lower(): _normalize_json_value(value) for key, value in item.items()}


def _serialize_item(item: dict) -> str:
    return json.dumps(item, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _extract_items(text: str) -> list | None:
    parsed = _try_parse_json(text)
    if not isinstance(parsed, dict):
        return None
    items = parsed.get("items")
    if isinstance(items, list):
        return items
    return None


def _is_valid_structure_prediction(text: str) -> bool:
    parsed = _try_parse_json(text)
    if not isinstance(parsed, dict):
        return False
    items = parsed.get("items")
    if not isinstance(items, list):
        return False
    for item in items:
        if not isinstance(item, dict):
            return False
        normalized_item = _normalize_item(item)
        if "value" not in normalized_item:
            return False
    return True


def _serialize_dim_value_signature(item: dict) -> str | None:
    normalized_item = _normalize_item(item)
    if "value" not in normalized_item:
        return None

    payload: dict[str, object] = {"value": normalized_item.get("value")}
    dims = normalized_item.get("dims")
    if dims is not None:
        if not isinstance(dims, dict):
            return None
        payload["has_dims"] = True
        payload["dims"] = dims
    else:
        payload["has_dims"] = False
    return _serialize_item(payload)


def _structure_alias_match(pred_items: list, gt_items: list) -> bool:
    if len(pred_items) != len(gt_items):
        return False
    if not all(isinstance(item, dict) for item in pred_items):
        return False
    if not all(isinstance(item, dict) for item in gt_items):
        return False

    gt_signatures = []
    pred_signatures = []
    for item in gt_items:
        sig = _serialize_dim_value_signature(item)
        if sig is None:
            return False
        gt_signatures.append(sig)
    for item in pred_items:
        sig = _serialize_dim_value_signature(item)
        if sig is None:
            return False
        pred_signatures.append(sig)
    return Counter(pred_signatures) == Counter(gt_signatures)


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
    target_kind = "structure"

    is_correct = False
    structure_exact_match = False
    structure_alias_value_match = False
    gt_items = _extract_items(ground_truth_text)
    pred_items = _extract_items(final_answer)
    pred_structure_valid = _is_valid_structure_prediction(final_answer)

    if isinstance(gt_items, list) and isinstance(pred_items, list):
        gt_dict_items = [item for item in gt_items if isinstance(item, dict)]
        pred_dict_items = [item for item in pred_items if isinstance(item, dict)]
        if len(gt_dict_items) == len(gt_items) and len(pred_dict_items) == len(pred_items):
            gt_counter = Counter(_serialize_item(_normalize_item(item)) for item in gt_dict_items)
            pred_counter = Counter(_serialize_item(_normalize_item(item)) for item in pred_dict_items)
            structure_exact_match = pred_counter == gt_counter
            structure_alias_value_match = _structure_alias_match(pred_dict_items, gt_dict_items)
            is_correct = structure_exact_match or structure_alias_value_match

    correctness_reward = 1.0 if is_correct else 0.0
    expected_table_name = task_info.get("table_name", "")

    expected_tables = _normalize_expected_tables(expected_table_name)
    sql_call_count, exp_table_sql_calls, exp_table_sql_success = _sql_call_stats(task_info, expected_tables)
    exp_table_hit_rate = _safe_ratio(exp_table_sql_calls, sql_call_count)
    exp_table_sql_succ_rate = _safe_ratio(exp_table_sql_success, exp_table_sql_calls)

    # Optional shaping for wrong answers: only when predicted answer format is valid,
    # and weighted by expected-table hit + SQL success ratios.
    shaping_bonus = 0.0
    if _SHAPING_ENABLED and not is_correct and pred_structure_valid and sql_call_count > 0 and exp_table_sql_calls > 0:
        shaping_factor = exp_table_hit_rate * exp_table_sql_succ_rate
        shaping_bonus = max(0.0, min(_MAX_SHAPING_BONUS, _MAX_SHAPING_BONUS * shaping_factor))

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
            "structure_exact_match": structure_exact_match,
            "structure_alias_value_match": structure_alias_value_match,
            "pred_structure_valid": float(pred_structure_valid),
            "ground_truth_item_count": len(gt_items) if isinstance(gt_items, list) else -1,
            "pred_item_count": len(pred_items) if isinstance(pred_items, list) else -1,
            "final_answer_extracted": final_answer,
        },
    )
