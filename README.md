# VAMPS: Visual-Assisted Mathematical Problem Solving Benchmark

Minimal reproduction package for the VAMPS benchmark code.

Dataset is available [on HuggingFace](https://huggingface.co/datasets/VAMPSBenchmark/VAMPS) ([![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)).

## Setup

```bash
conda create -n vamps python=3.10 -y
conda activate vamps
pip install -r requirements.txt
playwright install chromium
```

## Download full VAMPS dataset from Hugging Face

```bash
python download_vamps_dataset.py \
  --dataset-name VAMPSBenchmark/VAMPS \
  --split all \
  --out-dir data
```

This creates:

- `data/Konkour_EN/data.json` + `data/Konkour_EN/images/`
- `data/Konkour_FA/data.json` + `data/Konkour_FA/images/`
- `data/Synth_EN/data.json` + `data/Synth_EN/images/`
- `data/Synth_FA/data.json` + `data/Synth_FA/images/`

The script downloads all selected splits from Hugging Face parquet files and extracts image content into each split's `images/` directory.


Then point the scripts at one split file. For the baseline direct analytical (no-tool) run, use:

```bash
python run_api_no_tool.py \
  --model qwen/qwen3.5-27b \
  --api-provider openrouter \
  --api-key "$OPENROUTER_API_KEY" \
  --data data/Konkour_EN/data.json \
  --prompt PROMPT_BASELINE.md \
  --out results/no_tool.json
```

This writes an aggregated result JSON under `results/` and per-question responses in a model-specific subdirectory.

## Tool-using agent (Desmos)

Get a free Desmos API key from [Desmos API](https://www.desmos.com/my-api). Then run the command below:

```bash
python desmos_agent_benchmark.py \
  --input data/Konkour_EN/data.json \
  --output results/desmos_agent \
  --model-id qwen/qwen3.5-27b \
  --api-provider openrouter \
  --api-key "$OPENROUTER_API_KEY" \
  --desmos-api-key "$DESMOS_API_KEY"
```

This writes run outputs under `results/desmos_agent/`, including per-question traces and screenshots.

## Extract final option labels (with VLM-as-judge) + stats

```bash
python extract_answers.py \
   results/tool_agent/results_qwen-qwen3.5-27b.json \
  --output results/desmos_agent_extracted.json \
  --api-extraction \
  --api-provider openrouter \
  --api-model qwen/qwen3-vl-30b-a3b-instruct \
  --api-key "$OPENROUTER_API_KEY" \
  --stats-out results/desmos_agent_stats.json
```

This produces:
- `results/desmos_agent_extracted.json`: normalized final predictions per question.
- `results/desmos_agent_stats.json`: accuracy and extraction summary metrics.

## Suggested workflow

1. Download data (`download_vamps_dataset.py`).
2. Run baseline no-tool on one split.
3. Run Desmos tool-agent on the same split.
4. Run `extract_answers.py` on each run output.
5. Compare the resulting `*_stats.json` files.

For local endpoints, pass `--api-provider`, `--base-url`, and `--api-key` directly to the scripts.
