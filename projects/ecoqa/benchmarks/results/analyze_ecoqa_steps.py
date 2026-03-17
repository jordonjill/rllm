#!/usr/bin/env python3
"""Analyze EcoQA step traces from ecoqa_steps.jsonl."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?%?")
FULL_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?%?")


def _try_parse_json(value):
    if isinstance(value, (dict, list, int, float, bool)) or value is None:
        return value
    if not isinstance(value, str):
        return None

    stripped = value.strip()
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
        try:
            return json.loads(stripped[start : end + 1])
        except Exception:
            continue
    return None


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def _coerce_numeric_string(value: str) -> float | None:
    token = value.strip().replace(" ", "")
    if not token or not FULL_NUMBER_RE.fullmatch(token):
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
    if isinstance(value, (int, float)):
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
    return {str(k).strip().lower(): _normalize_json_value(v) for k, v in row.items()}


def _serialize_row(row: dict) -> str:
    return json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _serialize_value(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _serialize_row_values(row: dict) -> str:
    normalized = _normalize_row(row)
    tokens = sorted(_serialize_value(v) for v in normalized.values())
    return json.dumps(tokens, ensure_ascii=False, separators=(",", ":"))


def _extract_sql_query_and_last_rows(steps: list[dict]) -> tuple[str, list | None]:
    sql_query = ""
    sql_rows = None
    for step in steps:
        action = step.get("action", [])
        if isinstance(action, list):
            for call in action:
                fn = ((call or {}).get("function") or {}).get("name")
                if fn != "sql_query":
                    continue
                args = ((call.get("function") or {}).get("arguments")) or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                if isinstance(args, dict):
                    sql_query = str(args.get("query", sql_query))

        observation = step.get("observation", {})
        if isinstance(observation, dict) and "tool_outputs" in observation:
            for output in (observation.get("tool_outputs") or {}).values():
                parsed = _try_parse_json(output)
                if isinstance(parsed, list) and (parsed == [] or all(isinstance(x, dict) for x in parsed)):
                    sql_rows = parsed
    return sql_query, sql_rows


def _extract_final_answer(row: dict) -> str:
    steps = row.get("steps", [])
    if not steps:
        return ""
    metadata = ((steps[-1].get("info", {}) or {}).get("metadata", {}) or {})
    return str(metadata.get("final_answer_extracted", "") or "")


def _extract_pred_struct(final_answer: str) -> tuple[str, list | None, float | None]:
    parsed = _try_parse_json(final_answer)
    pred_type = "unknown"
    pred_rows = None
    pred_scalar = None

    if isinstance(parsed, dict):
        pred_type = str(parsed.get("type", "unknown")).strip().lower() or "unknown"
        for key in ("rows", "value", "data", "answer"):
            candidate = parsed.get(key)
            if isinstance(candidate, list):
                pred_rows = candidate
                break
        value = parsed.get("value")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            pred_scalar = float(value)
        elif isinstance(value, str):
            match = NUMBER_RE.search(value)
            if match:
                try:
                    pred_scalar = float(match.group(0).rstrip("%").replace(",", ""))
                except ValueError:
                    pass
    elif isinstance(parsed, list):
        pred_rows = parsed
    elif isinstance(parsed, (int, float)) and not isinstance(parsed, bool):
        pred_scalar = float(parsed)
    elif isinstance(parsed, str):
        match = NUMBER_RE.search(parsed)
        if match:
            try:
                pred_scalar = float(match.group(0).rstrip("%").replace(",", ""))
            except ValueError:
                pass
    return pred_type, pred_rows, pred_scalar


def _infer_target_kind(row: dict) -> str:
    question_type = str(row.get("question_type", "")).strip().lower()
    answer_type = str(row.get("answer_type", "")).strip().lower()
    if question_type == "single_table_error":
        return "no_data"
    if answer_type in {"list", "scalar"}:
        return answer_type
    return "unknown"


def _row_is_correct(row: dict) -> bool:
    steps = row.get("steps", [])
    if steps:
        info = (steps[-1].get("info", {}) or {})
        if "is_correct" in info:
            return bool(info.get("is_correct"))
        metadata = info.get("metadata", {})
        if isinstance(metadata, dict):
            correctness_reward = metadata.get("correctness_reward")
            if correctness_reward is not None:
                try:
                    return float(correctness_reward) >= 1.0
                except (TypeError, ValueError):
                    pass
    # Fallback for legacy rows without explicit correctness info.
    return float(row.get("reward", 0.0) or 0.0) >= 1.0


def _categorize_failure(row: dict) -> tuple[str, dict]:
    final_answer = _extract_final_answer(row)
    pred_type, pred_rows, pred_scalar = _extract_pred_struct(final_answer)
    target_kind = _infer_target_kind(row)
    gt = _try_parse_json(row.get("ground_truth", ""))
    sql_query, sql_rows = _extract_sql_query_and_last_rows(row.get("steps", []))

    category = "unknown_failure"
    if target_kind == "list":
        if pred_type != "list" and not isinstance(pred_rows, list):
            category = "format_error_list_to_non_list"
        elif sql_rows == []:
            category = "sql_empty_result"
        elif not isinstance(gt, list) or not isinstance(pred_rows, list):
            category = "parse_or_schema_error"
        else:
            gt_dict_rows = [x for x in gt if isinstance(x, dict)]
            pred_dict_rows = [x for x in pred_rows if isinstance(x, dict)]
            if len(gt_dict_rows) != len(gt) or len(pred_dict_rows) != len(pred_rows):
                category = "rows_not_object"
            else:
                gt_keys = set().union(*[set(_normalize_row(x).keys()) for x in gt_dict_rows]) if gt_dict_rows else set()
                pred_keys = set().union(*[set(_normalize_row(x).keys()) for x in pred_dict_rows]) if pred_dict_rows else set()
                gt_exact = Counter(_serialize_row(_normalize_row(x)) for x in gt_dict_rows)
                pred_exact = Counter(_serialize_row(_normalize_row(x)) for x in pred_dict_rows)
                gt_values = Counter(_serialize_row_values(x) for x in gt_dict_rows)
                pred_values = Counter(_serialize_row_values(x) for x in pred_dict_rows)

                if gt_exact == pred_exact and len(gt_exact) > 0:
                    category = "value_mismatch"
                elif gt_values == pred_values and len(gt_values) > 0:
                    category = "alias_only_mismatch"
                elif pred_keys != gt_keys:
                    missing = gt_keys - pred_keys
                    extra = pred_keys - gt_keys
                    if missing and extra:
                        category = "column_mismatch_plus_missing"
                    elif missing:
                        category = "missing_columns_or_incomplete_structure"
                    else:
                        category = "column_mismatch"
                else:
                    category = "value_mismatch"
    elif target_kind == "scalar":
        if pred_type == "list" or isinstance(pred_rows, list):
            category = "format_error_scalar_to_list"
        elif sql_rows == []:
            category = "sql_empty_result"
        else:
            gt_num = None
            if isinstance(gt, (int, float)) and not isinstance(gt, bool):
                gt_num = float(gt)
            elif isinstance(gt, str):
                match = NUMBER_RE.search(gt)
                if match:
                    try:
                        gt_num = float(match.group(0).rstrip("%").replace(",", ""))
                    except ValueError:
                        pass
            if pred_scalar is None:
                category = "scalar_not_parseable"
            elif gt_num is None:
                category = "scalar_text_or_schema_mismatch"
            else:
                tol = max(1e-4, 1e-3 * max(abs(gt_num), 1.0))
                category = "value_mismatch" if abs(pred_scalar - gt_num) > tol else "other_scalar_failure"
    elif target_kind == "no_data":
        category = "no_data_not_returned"

    return category, {
        "question_id": str(row.get("question_id", "")),
        "question": str(row.get("question", "")),
        "final_answer": final_answer[:220],
        "sql_query": sql_query[:220],
    }


def analyze(input_path: Path, top_n_examples: int = 3) -> dict:
    rows = [json.loads(line) for line in input_path.open(encoding="utf-8") if line.strip()]
    total = len(rows)
    success = sum(1 for row in rows if _row_is_correct(row))
    failure = total - success

    answer_type_stats = defaultdict(lambda: {"total": 0, "success": 0})
    question_type_stats = defaultdict(lambda: {"total": 0, "success": 0})
    step_distribution = Counter()
    termination_distribution = Counter()
    sql_call_distribution = Counter()

    for row in rows:
        ok = _row_is_correct(row)
        answer_type = str(row.get("answer_type", "unknown") or "unknown")
        question_type = str(row.get("question_type", "unknown") or "unknown")

        answer_type_stats[answer_type]["total"] += 1
        answer_type_stats[answer_type]["success"] += int(ok)
        question_type_stats[question_type]["total"] += 1
        question_type_stats[question_type]["success"] += int(ok)

        step_distribution[int(row.get("num_steps", 0) or 0)] += 1
        termination_distribution[str(row.get("termination_reason_inferred", ""))] += 1

        sql_calls = 0
        for step in row.get("steps", []):
            action = step.get("action", [])
            if not isinstance(action, list):
                continue
            for call in action:
                if ((call or {}).get("function") or {}).get("name") == "sql_query":
                    sql_calls += 1
        sql_call_distribution[sql_calls] += 1

    failure_categories = Counter()
    examples = defaultdict(list)
    for row in rows:
        if _row_is_correct(row):
            continue
        category, payload = _categorize_failure(row)
        failure_categories[category] += 1
        if len(examples[category]) < top_n_examples:
            examples[category].append(payload)

    return {
        "total": total,
        "success": success,
        "failure": failure,
        "success_rate": (success / total) if total else 0.0,
        "mean_reward": (sum(float(row.get("reward", 0.0) or 0.0) for row in rows) / total) if total else 0.0,
        "answer_type_stats": answer_type_stats,
        "question_type_stats": question_type_stats,
        "step_distribution": step_distribution,
        "termination_distribution": termination_distribution,
        "sql_call_distribution": sql_call_distribution,
        "failure_categories": failure_categories,
        "examples": examples,
    }


def _print_report(report: dict) -> None:
    print("=== OVERALL ===")
    print(
        json.dumps(
            {
                "total": report["total"],
                "success": report["success"],
                "failure": report["failure"],
                "success_rate_pct": round(report["success_rate"] * 100, 2),
                "mean_reward": round(report["mean_reward"], 4),
                "step_distribution": dict(report["step_distribution"]),
                "termination_distribution": dict(report["termination_distribution"]),
                "sql_call_distribution": dict(report["sql_call_distribution"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    print("\n=== BY ANSWER TYPE ===")
    for key in sorted(report["answer_type_stats"]):
        row = report["answer_type_stats"][key]
        rate = (row["success"] / row["total"] * 100) if row["total"] else 0.0
        print(f"{key}: {row['success']}/{row['total']} ({rate:.1f}%)")

    print("\n=== BY QUESTION TYPE ===")
    for key in sorted(report["question_type_stats"]):
        row = report["question_type_stats"][key]
        rate = (row["success"] / row["total"] * 100) if row["total"] else 0.0
        print(f"{key}: {row['success']}/{row['total']} ({rate:.1f}%)")

    print("\n=== FAILURE CATEGORIES ===")
    total_fail = max(report["failure"], 1)
    for key, value in report["failure_categories"].most_common():
        print(f"{key}: {value}/{report['failure']} ({value / total_fail * 100:.1f}%)")

    print("\n=== EXAMPLES ===")
    for key, _ in report["failure_categories"].most_common(6):
        print(f"-- {key}")
        for payload in report["examples"][key]:
            print(json.dumps(payload, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze EcoQA step traces.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).with_name("ecoqa_steps.jsonl"),
        help="Path to ecoqa_steps.jsonl",
    )
    parser.add_argument("--top-n-examples", type=int, default=3, help="How many examples per category to print.")
    parser.add_argument("--save-json", type=Path, default=None, help="Optional path to save JSON summary.")
    args = parser.parse_args()

    report = analyze(args.input, top_n_examples=args.top_n_examples)
    _print_report(report)

    if args.save_json is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        serializable = {
            **report,
            "answer_type_stats": dict(report["answer_type_stats"]),
            "question_type_stats": dict(report["question_type_stats"]),
            "step_distribution": dict(report["step_distribution"]),
            "termination_distribution": dict(report["termination_distribution"]),
            "sql_call_distribution": dict(report["sql_call_distribution"]),
            "failure_categories": dict(report["failure_categories"]),
            "examples": {k: v for k, v in report["examples"].items()},
        }
        with args.save_json.open("w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
        print(f"\nSaved JSON summary: {args.save_json}")


if __name__ == "__main__":
    main()
