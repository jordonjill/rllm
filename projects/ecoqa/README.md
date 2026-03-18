# EcoQA Single-Agent Training Framework

EcoQA mirrors the FinQA project structure, but focuses on **single-table economic QA** over CSV tables in `projects/ecoqa/data/csv`.

## Core Design

`EcoQAAgent` uses 4 tools:

1. `get_table_names`: list available tables
2. `get_table_info`: inspect schema and sample values
3. `sql_query`: run read-only SQL on in-memory SQLite tables
4. `calculator`: evaluate arithmetic expressions

All CSV tables are preloaded into one in-memory SQLite database, one SQL table per CSV.

## What `run_ecoqa.py` does

`python -m projects.ecoqa.run_ecoqa` is a **full inference-eval chain** for one model:

1. load/register `ecoqa` test split
2. run agent + tool environment end-to-end
3. score with EcoQA reward
4. print pass@k metrics

It is not training; it is for end-to-end evaluation/inference.

## Answer / Reward Format

- QA ground truth now uses a unified structured JSON format:
  - `{"rows":[{"result": ..., "year": ..., "month": ...}, ...]}`
  - no-data answer: `{"rows":[]}`
- Reward correctness compares `rows` structurally (order-insensitive) with normalized key/value matching.
- Recommended output convention:
  - put main target value in `result`
  - put qualifiers in other columns (`year/month/geo_name/...`)
- Reward metadata keeps only 5 fields:
  - `final_reward`
  - `correctness_reward`
  - `shaping_bonus`
  - `exp_table_hit_rate`
  - `exp_table_sql_succ_rate`

## Quick Start

1. Install project dependencies from root:

```bash
uv pip install -e ".[dev]"
```

2. Regenerate QA pairs from YAML and then register datasets:

```bash
python3 projects/ecoqa/data/generate_qa_pairs_from_yaml.py
```

By default this script reads `projects/ecoqa/data/yaml/*.yaml`, applies a reproducible random split (seed=20260318), and writes:

- `projects/ecoqa/data/qa_pairs/train_ecoqa.csv` (720)
- `projects/ecoqa/data/qa_pairs/val_ecoqa.csv` (90)
- `projects/ecoqa/data/qa_pairs/test_ecoqa.csv` (90)

Then register these CSV files:

```bash
python -m projects.ecoqa.prepare_ecoqa_data
```

3. Run single-model evaluation:

```bash
python -m projects.ecoqa.run_ecoqa
```

Save full per-step traces to JSONL (tools/actions/outputs per step):

```bash
ECOQA_SAVE_STEPS_JSONL=projects/ecoqa/benchmarks/results/ecoqa_steps.jsonl \
python -m projects.ecoqa.run_ecoqa
```

`run_ecoqa.py` currently reads runtime config from env vars (not CLI args).

Run current base model (default path in `run_ecoqa.py`):

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /root/autodl-tmp/models/Qwen3-4B-Instruct-2507 \
  --host 0.0.0.0 \
  --port 30000 \
  --dtype bfloat16

python -m projects.ecoqa.run_ecoqa
```

Run trained checkpoint:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /root/autodl-tmp/checkpoints/rllm-agent/ecoqa-4b \
  --host 0.0.0.0 \
  --port 30000 \
  --dtype bfloat16

ECOQA_MODEL_SOURCE=ckpt \
ECOQA_CKPT_MODEL_PATH=/root/autodl-tmp/checkpoints/rllm-agent/ecoqa-4b \
ECOQA_BASE_URL=http://127.0.0.1:30000/v1 \
ECOQA_TOKENIZER_MODEL=/root/autodl-tmp/models/Qwen3-4B-Instruct-2507 \
ECOQA_API_KEY=EMPTY \
python -m projects.ecoqa.run_ecoqa
```

If your model path or endpoint differs from the defaults in `run_ecoqa.py`,
override `ECOQA_BASE_MODEL_PATH`, `ECOQA_CKPT_MODEL_PATH`, and `ECOQA_BASE_URL`.

4. Run GRPO training:

```bash
bash projects/ecoqa/train_ecoqa.sh
```

## Model Comparison (Baseline / Trained / Online API)

Files are under `projects/ecoqa/benchmarks`:

- `model_profiles.json`: baseline model, trained model, and online API model profiles
- `results/model_comparison.csv`: aggregated comparison table
- `results/*_details.jsonl`: per-trajectory details per run

Run comparison:

```bash
python -m projects.ecoqa.run_ecoqa_benchmark \
  --split test \
  --repeat-n 1
```

Tips:

- Enable/disable models in `projects/ecoqa/benchmarks/model_profiles.json`.
- For online API models, set the env var referenced by `api_key_env` (for example `OPENAI_API_KEY`).
- For trained checkpoints, update `model` in the `trained_ecoqa_local` profile and start a serving endpoint.
- `model_comparison.csv` includes mean values for `final_reward`, `correctness_reward`, `shaping_bonus`, `exp_table_hit_rate`, and `exp_table_sql_succ_rate`.

## Tests

EcoQA-specific tests are in `projects/ecoqa/tests`:

- `test_reward_function.py`
- `test_tools_basic.py`

Run:

```bash
pytest -q projects/ecoqa/tests
```

## Data Layout

- Raw tables: `projects/ecoqa/data/csv/*.csv`
- QA YAML source: `projects/ecoqa/data/yaml/*.yaml`
- QA pairs: `projects/ecoqa/data/qa_pairs/*.csv`
