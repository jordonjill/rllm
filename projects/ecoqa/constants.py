import os
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
PROMPTS_DIR = PROJECT_DIR / "prompts"
QA_PAIRS_DIR = DATA_DIR / "qa_pairs"

# Table root for tool runtime
TABLES_ROOT = Path(os.getenv("ECOQA_TABLES_ROOT", str(DATA_DIR / "csv")))

# Dataset CSVs generated for RL
TRAIN_QUESTIONS_PATH = QA_PAIRS_DIR / "train_ecoqa.csv"
VAL_QUESTIONS_PATH = QA_PAIRS_DIR / "val_ecoqa.csv"
TEST_QUESTIONS_PATH = QA_PAIRS_DIR / "test_ecoqa.csv"

# Prompts
REACT_SYSTEM_PROMPT_PATH = PROMPTS_DIR / "react_system_prompt.txt"
QUESTION_GENERATION_PROMPT_PATH = PROMPTS_DIR / "question_generation_prompt.txt"
QUESTION_VERIFICATION_PROMPT_PATH = PROMPTS_DIR / "question_verification_prompt.txt"
