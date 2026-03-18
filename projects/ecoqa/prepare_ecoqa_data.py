import json

import pandas as pd
from rllm.data.dataset import DatasetRegistry
from .constants import TEST_QUESTIONS_PATH, TRAIN_QUESTIONS_PATH, VAL_QUESTIONS_PATH


def _load_csv(path):
    return pd.read_csv(path)


def _safe_str(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def _safe_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, float) and pd.isna(value):
        return False
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off", ""}:
        return False
    return False


def _normalize_structure_answer(value, question_id: str) -> str:
    raw = _safe_str(value)
    if not raw:
        raise ValueError(f"Missing answer for question_id={question_id}")

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid answer JSON for question_id={question_id}: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError(f"Answer must be JSON object for question_id={question_id}")

    items = parsed.get("items")
    if not isinstance(items, list):
        raise ValueError(f"Answer must contain list field 'items' for question_id={question_id}")
    if not all(isinstance(item, dict) for item in items):
        raise ValueError(f"Each item in answer.items must be object for question_id={question_id}")

    # Canonical JSON string for stable downstream parsing/comparison.
    return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))


def _ensure_qa_files_exist() -> None:
    missing_paths = [path for path in (TRAIN_QUESTIONS_PATH, VAL_QUESTIONS_PATH, TEST_QUESTIONS_PATH) if not path.exists()]
    if not missing_paths:
        return

    missing_text = ", ".join(str(path) for path in missing_paths)
    raise FileNotFoundError(
        f"EcoQA QA pair files are missing: {missing_text}. "
        "This project is configured to use pre-generated QA pairs only."
    )


def _patch_verl_data_source_column(dataset) -> None:
    if not hasattr(dataset, "get_verl_data_path"):
        return
    verl_path = dataset.get_verl_data_path()
    if not verl_path:
        return

    try:
        df = pd.read_parquet(verl_path)
    except Exception:
        return

    if "data_source" in df.columns:
        return
    if "extra_info" not in df.columns:
        return

    def _extract_data_source(extra_info):
        if isinstance(extra_info, dict):
            value = extra_info.get("data_source", "")
            if value is None:
                return "unknown"
            text = str(value).strip()
            return text or "unknown"
        return "unknown"

    df["data_source"] = df["extra_info"].apply(_extract_data_source)
    df.to_parquet(verl_path, index=False)


def prepare_ecoqa_data(force_regenerate: bool = False):
    if force_regenerate:
        print("force_regenerate=True is ignored: using pre-generated EcoQA QA pairs.")

    _ensure_qa_files_exist()

    train_df = _load_csv(TRAIN_QUESTIONS_PATH)
    val_df = _load_csv(VAL_QUESTIONS_PATH)
    test_df = _load_csv(TEST_QUESTIONS_PATH)

    def preprocess_fn(example):
        question_text = _safe_str(example.get("question")) or _safe_str(example.get("user_query"))
        question_id = _safe_str(example.get("id"))
        ground_truth = _normalize_structure_answer(example.get("answer"), question_id)
        return {
            "question": question_text,
            "ground_truth": ground_truth,
            "data_source": "ecoqa",
            "question_id": question_id,
            "table_name": _safe_str(example.get("table_name")),
            "ground_truth_sql": _safe_str(example.get("ground_truth_sql")),
            "requires_calculator": _safe_bool(example.get("requires_calculator")),
        }

    train_processed = [preprocess_fn(row) for _, row in train_df.iterrows()]
    val_processed = [preprocess_fn(row) for _, row in val_df.iterrows()]
    test_processed = [preprocess_fn(row) for _, row in test_df.iterrows()]

    train_dataset = DatasetRegistry.register_dataset("ecoqa", train_processed, "train")
    val_dataset = DatasetRegistry.register_dataset("ecoqa", val_processed, "val")
    test_dataset = DatasetRegistry.register_dataset("ecoqa", test_processed, "test")

    # Ensure val metrics are grouped under "ecoqa" instead of "unknown".
    for ds in (train_dataset, val_dataset, test_dataset):
        _patch_verl_data_source_column(ds)

    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, val_dataset, test_dataset = prepare_ecoqa_data()
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Train dataset path: {train_dataset.get_data_path()}")
    print(f"Validation dataset path: {val_dataset.get_data_path()}")
    print(f"Test dataset path: {test_dataset.get_data_path()}")
