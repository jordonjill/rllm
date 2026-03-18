#!/usr/bin/env python3
"""Generate EcoQA QA-pair CSV splits from YAML question sources."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

CSV_COLUMNS = [
    "id",
    "source_id",
    "table_name",
    "question",
    "ground_truth_sql",
    "answer",
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


def _normalize_signature_text(value: Any) -> str:
    text = _normalize_text(value)
    return re.sub(r"\s+", " ", text)


def _signature_key(example: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        _normalize_signature_text(example.get("table_name", "")),
        _normalize_signature_text(example.get("question", "")),
        _normalize_signature_text(example.get("ground_truth_sql", "")),
        _normalize_signature_text(example.get("answer", "")),
    )


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
                answer = _normalize_answer(raw_answer)
                ground_truth_sql = ""
                requires_calculator = False
            elif single_sql:
                answer = _normalize_answer(raw_answer)
                ground_truth_sql = single_sql
                requires_calculator = bool(question_item.get("requires_calculator", False))
            elif sql_1 and sql_2:
                answer = _normalize_answer(raw_answer)
                ground_truth_sql = json.dumps({"sql1": sql_1, "sql2": sql_2}, ensure_ascii=False)
                requires_calculator = bool(question_item.get("requires_calculator", True))
            else:
                raise ValueError(f"Question has no valid SQL/error annotation: {yaml_path} -> {question_item}")

            examples.append(
                {
                    "id": "",
                    "source_id": _normalize_text(question_item.get("id", "")),
                    "table_name": table_name,
                    "question": question_text,
                    "ground_truth_sql": ground_truth_sql,
                    "answer": answer,
                    "requires_calculator": requires_calculator,
                    "source_yaml": yaml_path.name,
                }
            )

    return examples


def _split_one_source_group(
    group_items: list[dict[str, Any]],
    *,
    n_train: int,
    n_val: int,
    n_test: int,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    buckets: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in group_items:
        buckets[_signature_key(item)].append(item)

    bucket_keys = list(buckets.keys())
    rng.shuffle(bucket_keys)

    split_rows: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    remaining = {"train": n_train, "val": n_val, "test": n_test}
    split_priority = {"train": 0, "val": 1, "test": 2}

    for key in bucket_keys:
        bucket = buckets[key]
        size = len(bucket)

        # Prefer splits that can absorb the full bucket while keeping target
        # sizes close; if none can, place in the split with largest remaining.
        feasible = [name for name in ("train", "val", "test") if remaining[name] >= size]
        if feasible:
            chosen = sorted(feasible, key=lambda name: (-remaining[name], split_priority[name]))[0]
        else:
            chosen = sorted(("train", "val", "test"), key=lambda name: (-remaining[name], split_priority[name]))[0]

        split_rows[chosen].extend(bucket)
        remaining[chosen] -= size

    return split_rows["train"], split_rows["val"], split_rows["test"]


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

        group_train, group_val, group_test = _split_one_source_group(
            group_items,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            rng=rng,
        )
        train.extend(group_train)
        val.extend(group_val)
        test.extend(group_test)

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
    print(f"{split_name}: {len(rows)} rows")


def _cross_split_signature_overlap(train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]]) -> int:
    train_signatures = {_signature_key(row) for row in train_rows}
    val_signatures = {_signature_key(row) for row in val_rows}
    test_signatures = {_signature_key(row) for row in test_rows}
    return len((train_signatures & val_signatures) | (train_signatures & test_signatures) | (val_signatures & test_signatures))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate EcoQA train/val/test CSV files from YAML question banks.")
    parser.add_argument("--yaml-dir", type=Path, default=Path("projects/ecoqa/data/yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("projects/ecoqa/data/qa_pairs"))
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260318)
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
    print(f"cross_split_duplicate_signatures: {_cross_split_signature_overlap(train_rows, val_rows, test_rows)}")


if __name__ == "__main__":
    main()
