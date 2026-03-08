# EcoQA Single-Agent Training Framework

This project mirrors the FinQA layout but targets **single-table numerical QA** over the CSV tables under `projects/ecoqa/data/csv`.

It is intentionally a first runnable framework version so we can iterate in later steps on:
- QA pair quality
- SQL difficulty
- reward shaping
- training hyperparameters

## Core Design

`EcoQAAgent` is a single agent with 4 tools:

1. `get_table_names`: list available tables
2. `get_table_info`: inspect columns/types/samples for one table
3. `sql_query`: execute SQL on in-memory SQLite tables
4. `calculator`: evaluate arithmetic expressions

## Important Note (FinQA comparison)

In FinQA, `sql_query` is a **SQL execution tool** (not a SQL generation tool). The model itself writes SQL and calls the execution tool.

This EcoQA framework follows the same pattern because you requested a single-agent + multi-tool setup.

## Quick Start

1. Install dependencies:

```bash
uv pip install -r projects/ecoqa/requirements.txt
```

2. Generate QA pairs (LLM + prompts + real SQL execution):

```bash
python -m projects.ecoqa.scripts.data_generation.generate_qa_pairs \
  --mode llm \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --base_url http://localhost:30000/v1
```

3. Register datasets:

```bash
python -m projects.ecoqa.prepare_ecoqa_data
```

4. Run inference:

```bash
python -m projects.ecoqa.run_ecoqa
```

5. Train (verl backend baseline):

```bash
bash projects/ecoqa/train_ecoqa.sh
```

## Current Scope

This version is a baseline framework with LLM-based QA generation and deterministic reward.
We can next iterate on harder SQL patterns, stricter rewards, and better training settings.

## Data Layout

- Raw tables: `projects/ecoqa/data/csv/*.csv`
- Generated QA pairs: `projects/ecoqa/data/qa_pairs/*.csv`
