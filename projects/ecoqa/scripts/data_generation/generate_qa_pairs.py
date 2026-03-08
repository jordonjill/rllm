"""Convert EcoQA YAML question files into train/val/test CSV splits.

Each of the 15 ``data/yaml/<table>_qa.yaml`` files contains up to 50 hand-crafted
questions with SQL ground truth.  This script:

1. Loads every YAML file.
2. Keeps questions with ``answer_type in (None, "error")`` *and* ``answer is None``
   as **no-data sentinel** rows (``answer_type = "no_data"``, ``answer = "NO_DATA"``).
   These are questions 46–50 in every file: unanswerable by design (future dates,
   non-existent columns, impossible conditions).  Keeping them teaches the agent to
   say "data not found" rather than hallucinate an answer.
3. Drops questions whose ``answer_type`` is a concrete type (scalar/list/dict) but
   whose ``answer`` is still ``None`` — these are incomplete entries.
4. Serialises ``list`` / ``dict`` answers to JSON strings so that every ``answer``
   cell is a plain string.
5. Splits the questions from **each table** independently into train / val / test
   using a stable random seed, preserving the difficulty distribution.
6. Writes three CSV files to ``data/qa_pairs/``.
"""

import json
import math
import random
import re
from pathlib import Path

import pandas as pd
import yaml

from projects.ecoqa.constants import (
    DATA_DIR,
    QA_PAIRS_DIR,
    TEST_QUESTIONS_PATH,
    TRAIN_QUESTIONS_PATH,
    VAL_QUESTIONS_PATH,
)

YAML_DIR = DATA_DIR / "yaml"

# Reproducible split
RANDOM_SEED = 42

# Split proportions (must sum to 1.0)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# TEST_RATIO is the remainder: 1 - TRAIN_RATIO - VAL_RATIO = 0.15

# Column order written to every CSV split
OUTPUT_COLUMNS = [
    "id",
    "user_query",
    "question",
    "answer",
    "answer_type",
    "question_type",
    "table_name",
    "difficulty",
    "ground_truth_sql",
    "columns_used_json",
    "rows_used_json",
    "explanation",
]


# Ground-truth string written for no-data sentinel questions (Q46-50).
# The reward function checks whether the agent output signals "data not found"
# when it sees this as the ground truth.
NO_DATA_ANSWER = "NO_DATA"


def _serialize_answer(answer) -> str:
    """Return a plain-string representation of any answer value."""
    if isinstance(answer, (dict, list)):
        return json.dumps(answer, ensure_ascii=False)
    return str(answer)


def _load_yaml_file(yaml_path: Path) -> list[dict]:
    """Parse one YAML QA file and return a list of valid question dicts."""
    with open(yaml_path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    table_name: str = data.get("table_name") or re.sub(r"_(qa|questions)$", "", yaml_path.stem)
    rows: list[dict] = []

    for q in data.get("questions", []):
        answer = q.get("answer")
        answer_type = q.get("answer_type")

        # No-data sentinel questions: answer is None AND answer_type is either absent
        # (None — the question was written without a type) or explicitly "error" (the
        # question was flagged as unanswerable).  Both signal the same intent: the data
        # does not exist in the database.  These are the H. 边界与容错 questions (Q46-50)
        # in every table: future dates, non-existent columns, impossible conditions.
        # We keep them as no_data rows to teach the agent to say "data not found".
        if answer is None and answer_type in (None, "error"):
            rows.append(
                {
                    "user_query": q["question"],
                    "question": q["question"],
                    "answer": NO_DATA_ANSWER,
                    "answer_type": "no_data",
                    "question_type": "no_data",
                    "table_name": table_name,
                    "difficulty": q.get("difficulty", "medium"),
                    "ground_truth_sql": "",
                    "columns_used_json": "[]",
                    "rows_used_json": "[]",
                    "explanation": q.get("expected_error", ""),
                }
            )
            continue

        # Drop genuinely incomplete questions: answer_type is set but answer is missing.
        if answer is None:
            continue

        columns_used: list = q.get("columns_used") or []

        rows.append(
            {
                "user_query": q["question"],
                "question": q["question"],
                "answer": _serialize_answer(answer),
                "answer_type": answer_type or "scalar",
                "question_type": answer_type or "scalar",
                "table_name": table_name,
                "difficulty": q.get("difficulty", "medium"),
                "ground_truth_sql": q.get("sql", ""),
                "columns_used_json": json.dumps(columns_used, ensure_ascii=False),
                "rows_used_json": "[]",
                "explanation": q.get("answer_unit", ""),
            }
        )

    return rows


def _split_questions(
    questions: list[dict],
    *,
    seed: int = RANDOM_SEED,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Stratified train/val/test split that respects *difficulty* distribution.

    Questions in each difficulty bucket are shuffled independently so that
    every split gets a representative sample of easy, medium, and hard items.
    """
    by_difficulty: dict[str, list[dict]] = {}
    for q in questions:
        bucket = q.get("difficulty", "medium")
        by_difficulty.setdefault(bucket, []).append(q)

    rng = random.Random(seed)
    train, val, test = [], [], []

    for bucket_qs in by_difficulty.values():
        shuffled = list(bucket_qs)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = round(n * TRAIN_RATIO)
        n_val = round(n * VAL_RATIO)
        # Clamp to ensure counts stay within bounds; remainder goes to test
        n_train = min(n_train, n)
        n_val = min(n_val, max(0, n - n_train))
        train.extend(shuffled[:n_train])
        val.extend(shuffled[n_train : n_train + n_val])
        test.extend(shuffled[n_train + n_val :])

    return train, val, test


def _finalise_split(rows: list[dict]) -> pd.DataFrame:
    """Assign sequential IDs and enforce column order."""
    df = pd.DataFrame(rows).reset_index(drop=True)
    df["id"] = df.index
    # Keep only declared columns (in order), skipping any that are absent
    cols = [c for c in OUTPUT_COLUMNS if c in df.columns]
    return df[cols]


def generate_qa_pairs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load YAML files, split, and write CSV splits.

    Returns
    -------
    train_df, val_df, test_df : pd.DataFrame
    """
    if not YAML_DIR.exists():
        raise FileNotFoundError(f"YAML directory not found: {YAML_DIR}")

    yaml_files = sorted(YAML_DIR.glob("*.yaml"))
    if not yaml_files:
        raise FileNotFoundError(f"No YAML files found in {YAML_DIR}")

    train_rows: list[dict] = []
    val_rows: list[dict] = []
    test_rows: list[dict] = []

    for yaml_path in yaml_files:
        questions = _load_yaml_file(yaml_path)
        if not questions:
            print(f"  [warn] No valid questions in {yaml_path.name}")
            continue

        t, v, te = _split_questions(questions)
        train_rows.extend(t)
        val_rows.extend(v)
        test_rows.extend(te)
        print(f"  {yaml_path.name}: {len(t)} train / {len(v)} val / {len(te)} test")

    train_df = _finalise_split(train_rows)
    val_df = _finalise_split(val_rows)
    test_df = _finalise_split(test_rows)

    QA_PAIRS_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_QUESTIONS_PATH, index=False)
    val_df.to_csv(VAL_QUESTIONS_PATH, index=False)
    test_df.to_csv(TEST_QUESTIONS_PATH, index=False)

    print(f"Train : {len(train_df):4d} rows  → {TRAIN_QUESTIONS_PATH}")
    print(f"Val   : {len(val_df):4d} rows  → {VAL_QUESTIONS_PATH}")
    print(f"Test  : {len(test_df):4d} rows  → {TEST_QUESTIONS_PATH}")

    return train_df, val_df, test_df


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).parents[4]))
    generate_qa_pairs()
