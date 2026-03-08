import json

import pandas as pd

from rllm.data.dataset import DatasetRegistry

from .constants import TEST_QUESTIONS_PATH, TRAIN_QUESTIONS_PATH, VAL_QUESTIONS_PATH


def _load_csv(path):
    return pd.read_csv(path)


def _parse_json_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            parsed = stripped
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, str):
            cleaned = parsed.strip()
            return [cleaned] if cleaned else []
    return []


def _safe_str(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def _ensure_qa_files_exist() -> None:
    missing_paths = [path for path in (TRAIN_QUESTIONS_PATH, VAL_QUESTIONS_PATH, TEST_QUESTIONS_PATH) if not path.exists()]
    if not missing_paths:
        return

    missing_text = ", ".join(str(path) for path in missing_paths)
    raise FileNotFoundError(
        f"EcoQA QA pair files are missing: {missing_text}. "
        "This project is configured to use pre-generated QA pairs only."
    )


def prepare_ecoqa_data(force_regenerate: bool = False):
    if force_regenerate:
        print("force_regenerate=True is ignored: using pre-generated EcoQA QA pairs.")

    _ensure_qa_files_exist()

    train_df = _load_csv(TRAIN_QUESTIONS_PATH)
    val_df = _load_csv(VAL_QUESTIONS_PATH)
    test_df = _load_csv(TEST_QUESTIONS_PATH)

    def preprocess_fn(example):
        return {
            "question": _safe_str(example.get("user_query")),
            "ground_truth": _safe_str(example.get("answer")),
            "data_source": "ecoqa",
            "question_id": _safe_str(example.get("id")),
            "question_type": _safe_str(example.get("question_type")).lower(),
            "answer_type": _safe_str(example.get("answer_type")).lower(),
            "core_question": _safe_str(example.get("question")),
            "table_name": _safe_str(example.get("table_name")),
            "ground_truth_sql": _safe_str(example.get("ground_truth_sql")),
            "columns_used": _parse_json_list(example.get("columns_used_json")),
            "rows_used": _parse_json_list(example.get("rows_used_json")),
            "explanation": _safe_str(example.get("explanation")),
        }

    train_processed = [preprocess_fn(row) for _, row in train_df.iterrows()]
    val_processed = [preprocess_fn(row) for _, row in val_df.iterrows()]
    test_processed = [preprocess_fn(row) for _, row in test_df.iterrows()]

    train_dataset = DatasetRegistry.register_dataset("ecoqa", train_processed, "train")
    val_dataset = DatasetRegistry.register_dataset("ecoqa", val_processed, "val")
    test_dataset = DatasetRegistry.register_dataset("ecoqa", test_processed, "test")

    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, val_dataset, test_dataset = prepare_ecoqa_data()
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Train dataset path: {train_dataset.get_data_path()}")
    print(f"Validation dataset path: {val_dataset.get_data_path()}")
    print(f"Test dataset path: {test_dataset.get_data_path()}")
