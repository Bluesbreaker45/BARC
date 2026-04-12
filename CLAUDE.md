# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BARC (Bootstrapping ARC) generates 100k+ synthetic [ARC](https://arcprize.org/) problems using LLMs. ARC problems are grid transformation puzzles: given input/output grid pairs demonstrating a rule, predict the output for a new input. The system uses 162+ manually-written seed solutions as few-shot examples for LLM-based generation of new problems.

This project is developed under zsh and `BARC` conda environment.

Paper: [Combining Induction and Transduction for Abstract Reasoning](https://arxiv.org/abs/2411.02272)

## Three-Stage Generation Pipeline

Run the full pipeline via `data_generation_script.sh`:

1. **`generate_descriptions.py`** — Uses GPT-4 with self-instruct to generate natural language descriptions of new ARC problems from seed descriptions. Produces JSONL files in `generated_descriptions/`.
2. **`generate_code.py`** — Uses GPT-4o-mini with RAG (embedding-based retrieval of similar seeds) to generate Python code (`main()` + `generate_input()`) from descriptions. Produces JSONL in `generated_code/`.
3. **`generate_problems.py`** — Executes generated code in a sandbox to create concrete input/output grid pairs. Validates grids (2D numpy arrays, values 0-9, non-trivial, deterministic `main()`). Produces JSONL in `generated_problems/`.

## Key Commands

```bash
# Generate descriptions (batched, 10 per batch, 10 seed offsets)
python generate_descriptions.py --outdir generated_descriptions --batch_request --model gpt-4 --num_generations 10 --max_tokens 1024 --batch_size 1000 --num_descriptions 75 --rng_offset 0

# Generate code (with/without function suggestions for diversity)
python generate_code.py --outdir generated_code --batch_request --ignore_cache_samples --prompt_model gpt-4o-mini -n 16 -s 4 --nohtml --jsonl <descriptions.jsonl>
python generate_code.py --suggest_function --outdir generated_code --batch_request --ignore_cache_samples --prompt_model gpt-4o-mini -n 16 -s 4 --nohtml --jsonl <descriptions.jsonl>

# Generate problems from code
python generate_problems.py --jsonl <code.jsonl> --outdir generated_problems

# Evaluate induction code samples
python eval_code_samples.py

# Score inference results (requires pip install -r requirements.txt)
python evaluation.py

# Visualize generated problems as HTML
python visualize_problems.py --jsonl <problems.jsonl> --outdir generated_problems/visualized

# View a specific ARC problem
python view_problem.py
```

## Seed Problem Format

Each file in `seeds/` (named by ARC problem ID, e.g. `00d62c1b.py`) follows this structure:

```python
from common import *
import numpy as np

# concepts:
# <concept labels, e.g. topology, color mapping, pattern repetition>

# description:
# <natural language description of the transformation>

def main(input_grid):
    # Transformation: input grid -> output grid
    return output_grid

def generate_input():
    # Random valid input generator
    return input_grid

# ============= remove below this point for prompting =============
# (test/verification code below this line is stripped when building prompts)
```

## Architecture

### Core Modules

- **`seeds/common.py`** (~1500 lines) — Shared grid manipulation library used by all seeds. Key utilities: `Color` enum (0-9), `flood_fill`, `blit_sprite`, `crop`, `detect_objects`, connected components, symmetry detection (rotational/translational/mirror), sprite generation. Note: the Color class docstring intentionally lies to LLMs (says colors are strings) to prevent them from doing arithmetic on color values.
- **`llm.py`** — Multi-provider LLM client (OpenAI, Groq, DeepSeek, vLLM, OpenRouter) with disk caching, cost tracking, and batch API support. API keys via env vars: `OPENAI_API_KEY`, `GROQ_API_KEY`, `DEEPSEEK_API_KEY`, `OPENROUTER_API_KEY`.
- **`execution.py`** — Sandboxed code execution with import restrictions (blocks `os`, `sys`), 1-second timeout via `func_timeout`, and multiprocessing support (`multi_execute_transformation`, `multi_execute_input_generator`).
- **`utils.py`** — AST-based code parsing: `extract_functions`, `extract_function_calls`, `parse_code`, `remove_trailing_code`, HTML grid visualization.
- **`prompt.py`** — Prompt construction: `get_common_lib_from_file`, `prune_common_lib` (trims common.py to only functions used in seed examples for context efficiency).
- **`similarity.py`** — Embedding-based RAG for finding similar seed problems to use as few-shot examples during code generation.

### Grid Representation

Grids are 2D NumPy arrays with integer values 0-9 representing 10 colors: Black(0), Blue(1), Red(2), Green(3), Yellow(4), Grey(5), Pink(6), Orange(7), Teal(8), Maroon(9). Typical size: 1x1 to 30x30.

For finetuning prompts, grids are serialized as color name strings (e.g. "Gray Black Black Gray Black\nGray Black Black Gray Black").

### Datasets

- **`seeds/`** — 162+ manually-written ARC solutions (the ground truth seed set)
- **`seeds-inline/`** — Same seeds with `common.py` functions inlined (self-contained, no imports from common). Variables are prefixed with `__NNN__` to avoid name collisions. Recent work on `inline` branch.
- **`synthetic_problems/`** — LLM-generated problems (JSON + PNG + Python per problem)
- **`synthetic_problems-inline/`** — Inlined versions of synthetic problems
- **`ConceptARC/`** — Problems categorized by 17 visual concepts (AboveBelow, Center, Count, etc.)

### Finetuning (`finetune/`)

Uses HuggingFace alignment-handbook. Two approaches:
- **Induction** — Model generates solution code given ARC problem grids
- **Transduction** — Model directly outputs the test grid given problem grids

Models: Llama-3/3.1 (7B-70B). Inference via vLLM. See `finetune/alignment-handbook/recipes/barc/` for training configs.

## Environment Setup

Shell is zsh. The conda environment for this project is named **`BARC`** — activate it before running any script (or use `conda run -n BARC python ...`). It has `numpy`, `scipy`, `matplotlib`, `tqdm`, `orjsonl`, `func-timeout`, etc.

```zsh
conda activate BARC
# or for finetuning workflows:
# For finetuning: see README.md for alignment-handbook setup
```
