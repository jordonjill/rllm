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
        question_text = _safe_str(example.get("question")) or _safe_str(example.get("user_query"))
        return {
            "question": question_text,
            "ground_truth": _safe_str(example.get("answer")),
            "data_source": "ecoqa",
            "question_id": _safe_str(example.get("id")),
            "question_type": _safe_str(example.get("question_type")).lower(),
            "answer_type": _safe_str(example.get("answer_type")).lower(),
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

    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, val_dataset, test_dataset = prepare_ecoqa_data()
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Train dataset path: {train_dataset.get_data_path()}")
    print(f"Validation dataset path: {val_dataset.get_data_path()}")
    print(f"Test dataset path: {test_dataset.get_data_path()}")
