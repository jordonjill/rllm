"""Reward function for EcoQA.

Design rationale
----------------
We use **deterministic numeric comparison with partial credit** rather than an
LLM judge.  This keeps training fast, reproducible, and free of API calls while
still providing a richer signal than a purely binary reward:

* **scalar** questions (single number): binary 0 / 1 with relative ± 0.1 %
  tolerance (and percent-vs-decimal equivalence).
* **dict / list** questions (multi-value): *partial credit* equal to the
  fraction of ground-truth values that are matched in the prediction.  This
  avoids the "all-or-nothing" problem for ranked/sorted answers.
* **no_data** sentinel questions (Q46-50, unanswerable by design): the agent
  earns a reward of 1.0 **only** if it explicitly signals that the data is not
  available.  This discourages hallucination on out-of-range queries.
"""

import json
import re

from rllm.rewards.reward_types import RewardOutput

# ── Pattern constants ────────────────────────────────────────────────────────

_FINAL_ANSWER_CODE_BLOCK_RE = re.compile(r"```\s*FINAL ANSWER:\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_FINAL_ANSWER_PARAGRAPH_RE = re.compile(r"FINAL ANSWER:\s*(.*?)(?=\n\s*\n)", re.DOTALL | re.IGNORECASE)
_FINAL_ANSWER_TAIL_RE = re.compile(r"FINAL ANSWER:\s*(.*)$", re.DOTALL | re.IGNORECASE)
_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?%?")

# Sentinel string written to the CSV for unanswerable questions.
NO_DATA_ANSWER = "NO_DATA"

# Phrases (lower-case) that count as an explicit "no data" response.
_NO_DATA_PHRASES: tuple[str, ...] = (
    "no data",
    "no result",
    "data not found",
    "not found",
    "not available",
    "not in the database",
    "does not exist",
    "数据不存在",
    "查不到",
    "没有数据",
    "未找到",
    "无法找到",
    "数据超出范围",
    "不在数据范围",
    "没有该数据",
    "无数据",
    "数据为空",
    "找不到",
    "数据库中没有",
    "超出数据范围",
    "null",
    "none",
)

# ── Helpers ──────────────────────────────────────────────────────────────────


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


def _extract_numbers(text: str) -> list[float]:
    """Return all numbers found in *text* as a sorted list of floats.

    Percentage signs are stripped without dividing by 100, because EcoQA
    stores percentage-unit values as their face value (e.g. ``15.077`` means
    15.077 %).  An agent that writes ``15.077%`` should therefore match a
    ground-truth value of ``15.077``.
    """
    tokens = _NUMBER_RE.findall(text)
    results = []
    for token in tokens:
        clean = token.rstrip("%").replace(",", "")
        try:
            results.append(float(clean))
        except ValueError:
            pass
    return sorted(results)


def _numbers_close(predicted: float, ground_truth: float) -> bool:
    # Use both relative (0.1 %) and absolute (1e-4) tolerance so that
    # very small numbers (e.g. 0.001) are compared with a consistent
    # relative tolerance rather than only an absolute floor.
    rel_tol = 1e-3  # 0.1 %
    abs_tol = 1e-4
    return abs(predicted - ground_truth) <= max(abs_tol, rel_tol * max(abs(ground_truth), abs(predicted), 1e-9))


# ── Per-type scorers ─────────────────────────────────────────────────────────


def _score_no_data(pred_text: str) -> float:
    """Return 1.0 iff the prediction explicitly acknowledges data absence."""
    lower = pred_text.lower()
    return 1.0 if any(phrase in lower for phrase in _NO_DATA_PHRASES) else 0.0


def _score_scalar(pred_text: str, gt_str: str) -> float:
    """Exact numeric comparison with tolerance; handles percent equivalence."""
    pred_num, pred_pct = _parse_number(pred_text)
    gt_num, gt_pct = _parse_number(gt_str)

    if pred_num is not None and gt_num is not None:
        # Allow decimal vs percentage equivalence (e.g. 5.5% vs 0.055).
        if pred_pct and not gt_pct:
            pred_num = pred_num / 100.0
        if gt_pct and not pred_pct:
            gt_num = gt_num / 100.0
        return 1.0 if _numbers_close(pred_num, gt_num) else 0.0

    # Non-numeric scalar (e.g. city name) — normalised text match.
    return 1.0 if _normalize_text(pred_text) == _normalize_text(gt_str) else 0.0


def _score_json(pred_text: str, gt_text: str) -> float:
    """Partial-credit scoring for list / dict ground truth.

    Returns the fraction of ground-truth numeric values that appear
    (approximately) in the prediction.  Non-numeric JSON falls back to
    binary normalised text match.

    Partial credit gives the RL agent a smoother gradient: getting *some*
    values right earns a reward even when the full answer is incomplete.
    """
    try:
        json.loads(gt_text)  # validate JSON
    except (json.JSONDecodeError, ValueError):
        return 1.0 if _normalize_text(pred_text) == _normalize_text(gt_text) else 0.0

    gt_numbers = _extract_numbers(gt_text)
    if not gt_numbers:
        # Non-numeric JSON: binary text match.
        return 1.0 if _normalize_text(pred_text) == _normalize_text(gt_text) else 0.0

    pred_numbers = _extract_numbers(pred_text)

    # Greedy matching: for each GT number, consume the first close pred number.
    remaining = list(pred_numbers)
    matched = 0
    for gt_val in gt_numbers:
        idx = next((i for i, pv in enumerate(remaining) if _numbers_close(pv, gt_val)), None)
        if idx is not None:
            matched += 1
            remaining.pop(idx)

    return matched / len(gt_numbers)


# ── Table-access bookkeeping ─────────────────────────────────────────────────


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


# ── Main reward entry-point ──────────────────────────────────────────────────


def eco_qa_reward_function(task_info: dict, action: str) -> RewardOutput:
    """Compute the reward for one agent turn.

    Scoring rules
    -------------
    ground_truth == "NO_DATA"
        The question is unanswerable (Q46-50 sentinels).  Score 1.0 iff the
        agent explicitly states that the data is not available.
    ground_truth starts with '[' or '{'
        JSON list / dict.  Score = fraction of GT numbers found in prediction
        (partial credit, 0.0–1.0).
    otherwise
        Scalar / text.  Binary 0 or 1 with numeric tolerance.
    """
    question = task_info.get("question")
    ground_truth = task_info.get("ground_truth")

    if not action or not question or ground_truth in (None, ""):
        return RewardOutput(
            reward=0.0,
            is_correct=False,
            metadata={"correctness_reward": 0.0, "right_table_access_reward": 0.0, "answer_type": "unknown"},
        )

    final_answer = _extract_final_answer(action)
    gt_str = str(ground_truth).strip()

    if gt_str == NO_DATA_ANSWER:
        # Unanswerable sentinel question.
        score = _score_no_data(final_answer)
        answer_type = "no_data"
    elif gt_str.startswith(("[", "{")):
        # JSON list or dict — partial credit.
        score = _score_json(final_answer, gt_str)
        answer_type = "json"
    else:
        # Scalar (numeric or text).
        score = _score_scalar(final_answer, gt_str)
        answer_type = "scalar"

    # For reporting / early stopping, treat score >= 1.0 as "correct".
    is_correct = score >= 1.0

    accessed_tables = task_info.get("accessed_tables", [])
    expected_table_name = task_info.get("table_name", "")
    right_table_access_reward = _check_right_table_accessed(accessed_tables, expected_table_name)

    return RewardOutput(
        reward=score,
        is_correct=is_correct,
        metadata={
            "correctness_reward": score,
            "right_table_access_reward": right_table_access_reward,
            "final_answer_extracted": final_answer,
            "answer_type": answer_type,
        },
    )
