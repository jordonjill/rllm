#!/usr/bin/env python3
"""Generate EcoQA QA-pair CSV splits from YAML question sources."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

CSV_COLUMNS = [
    "id",
    "source_id",
    "table_name",
    "question_type",
    "question",
    "ground_truth_sql",
    "answer",
    "answer_type",
    "requires_calculator",
    "source_yaml",
]


def _normalize_sql(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_answer(value: Any) -> str:
    if value is None:
        value = {"items": []}
    if not isinstance(value, (dict, list)):
        raise ValueError(f"answer must be dict/list for structured format, got: {type(value).__name__}")
    return json.dumps(value, ensure_ascii=False)


def _load_examples(yaml_dir: Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []

    for yaml_path in sorted(yaml_dir.glob("*_qa.yaml")):
        payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid YAML payload: {yaml_path}")

        table_name = str(payload.get("table_name", yaml_path.stem.replace("_qa", ""))).strip()
        questions = payload.get("questions", [])
        if not isinstance(questions, list):
            raise ValueError(f"'questions' must be a list in {yaml_path}")

        for question_item in questions:
            if not isinstance(question_item, dict):
                raise ValueError(f"Question item must be object in {yaml_path}")

            question_text = str(question_item.get("question", "")).strip()
            if not question_text:
                raise ValueError(f"Missing question text in {yaml_path}")

            expected_error = question_item.get("expected_error")
            single_sql = _normalize_sql(question_item.get("sql"))
            sql_1 = _normalize_sql(question_item.get("sql1"))
            sql_2 = _normalize_sql(question_item.get("sql2"))
            raw_answer = question_item.get("answer")

            if expected_error is not None:
                question_type = "single_table_error"
                answer = _normalize_answer(raw_answer)
                answer_type = "structure"
                ground_truth_sql = ""
                requires_calculator = False
            elif single_sql:
                question_type = "single_table"
                answer = _normalize_answer(raw_answer)
                answer_type = "structure"
                ground_truth_sql = single_sql
                requires_calculator = bool(question_item.get("requires_calculator", False))
            elif sql_1 and sql_2:
                question_type = "single_table"
                answer = _normalize_answer(raw_answer)
                answer_type = "structure"
                ground_truth_sql = json.dumps({"sql1": sql_1, "sql2": sql_2}, ensure_ascii=False)
                requires_calculator = bool(question_item.get("requires_calculator", True))
            else:
                raise ValueError(f"Question has no valid SQL/error annotation: {yaml_path} -> {question_item}")

            examples.append(
                {
                    "id": "",
                    "source_id": _normalize_text(question_item.get("id", "")),
                    "table_name": table_name,
                    "question_type": question_type,
                    "question": question_text,
                    "ground_truth_sql": ground_truth_sql,
                    "answer": answer,
                    "answer_type": answer_type,
                    "requires_calculator": requires_calculator,
                    "source_yaml": yaml_path.name,
                }
            )

    return examples


def _split_examples(
    examples: list[dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Require ratios such that train_ratio > 0, val_ratio > 0, train_ratio + val_ratio < 1")

    rng = random.Random(seed)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in examples:
        grouped[item["source_yaml"]].append(item)

    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []

    for _, group_items in sorted(grouped.items()):
        rng.shuffle(group_items)
        n = len(group_items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val

        if n_test <= 0:
            n_test = 1
            if n_val > 1:
                n_val -= 1
            else:
                n_train -= 1

        train.extend(group_items[:n_train])
        val.extend(group_items[n_train : n_train + n_val])
        test.extend(group_items[n_train + n_val :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    cursor = 1
    for split_rows in (train, val, test):
        for row in split_rows:
            row["id"] = str(cursor)
            cursor += 1

    return train, val, test


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})


def _print_split_stats(split_name: str, rows: list[dict[str, Any]]) -> None:
    question_type_counts = Counter(row["question_type"] for row in rows)
    print(
        f"{split_name}: {len(rows)} rows | "
        f"single_table={question_type_counts.get('single_table', 0)} | "
        f"single_table_error={question_type_counts.get('single_table_error', 0)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate EcoQA train/val/test CSV files from YAML question banks.")
    parser.add_argument("--yaml-dir", type=Path, default=Path("projects/ecoqa/data/yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("projects/ecoqa/data/qa_pairs"))
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    examples = _load_examples(args.yaml_dir)
    train_rows, val_rows, test_rows = _split_examples(
        examples=examples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    _write_csv(args.output_dir / "train_ecoqa.csv", train_rows)
    _write_csv(args.output_dir / "val_ecoqa.csv", val_rows)
    _write_csv(args.output_dir / "test_ecoqa.csv", test_rows)

    print(f"Loaded examples: {len(examples)}")
    _print_split_stats("train", train_rows)
    _print_split_stats("val", val_rows)
    _print_split_stats("test", test_rows)


if __name__ == "__main__":
    main()
