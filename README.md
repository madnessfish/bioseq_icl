# A systematic evaluation of in-context learning in large language models for antibody characterization

A research project exploring the use of Large Language Models (LLMs) for antibody sequence analysis through in-context learning. This repository implements multiple scenarios for antibody classification tasks using few-shot and zero-shot prompting strategies.

## Overview
<p align="center">
  <img src="https://github.com/user-attachments/assets/dbe28915-8e10-4c5e-86e0-e98070debb92" width="100%">
</p>

This project investigates how LLMs can be applied to biological sequence classification tasks, specifically focusing on antibody-related predictions. The pipeline supports three main classification scenarios with various prompting strategies including zero-shot, few-shot, and Chain-of-Thought (CoT) reasoning.

## Data Access

Training and test datasets are available at the link specified in `data.txt`.

## Quick Start with Jupyter Notebook

For a quick hands-on demonstration of the complete pipeline, check out the interactive Jupyter notebook:

**`jupyter_notebook/prompt_test.ipynb`**

This notebook provides a step-by-step walkthrough of the entire ICL pipeline:

1. **Data Loading**: Load training and test datasets for antibody humanness prediction (Human vs. Rhesus)
2. **Demonstration Selection**: Select few-shot examples using similarity-based strategies (Levenshtein distance)
3. **Prompt Generation**: Create prompts with system instructions and few-shot examples
4. **LLM Inference**: Run prompts through GPT models with API integration
5. **Performance Evaluation**: Calculate accuracy metrics across multiple runs

**Key Features:**
- Complete end-to-end example for Scenario 1 (Antibody Humanness)
- Uses GPT-4o with temperature 0.2 and 5 repetitions per test instance
- Demonstrates data validation (no test set leakage)
- Includes response parsing and performance metrics
- Achieves ~93-94% accuracy on 100 test instances

**To run:**
```bash
cd jupyter_notebook
jupyter notebook prompt_test.ipynb
```

**Requirements:**
- Set `OPENAI_API_KEY` environment variable
- Install dependencies: `pandas`, `numpy`, `openai`, `python-Levenshtein`, `scikit-learn`

This notebook is ideal for understanding the complete workflow before running the full pipeline on larger datasets.

## Pipeline Workflow

### 1. Demonstration Selection (`00_demonstration_selection/`)

Select few-shot examples using various strategies:

- **Random sampling**: Random selection of positive and negative examples
- **Similar sampling**: Levenshtein distance-based selection
- **Similar (Balanced) sampling**: Ensures balanced representation of classes

**Example:**
```bash
python 00_demonstration_selection/00_demonstration.py \
  -d <input_dir> \
  -i <input_file> \
  -o <output_dir> \
  -m <mode>
```

**Modes:**
- `H`: Heavy chain only
- `HL`: Heavy and light chain
- `naiveness_mouse`: Mouse naiveness prediction
- `naiveness_rhesus`: Rhesus macaque naiveness prediction

### 2. Prompt Generation (`01_prompt_generation/`)

Generate prompts with selected demonstrations for LLM inference.

**Example:**
```bash
python 01_prompt_generation/scenario1_prompt_fewshot_ab_naiveness.py \
  -d <input_dir> \
  -i <demonstration_file> \
  -t <test_file> \
  -o <output_dir> \
  -m <mode>
```

### 3. Model Inference (`02_run_prompt/`)

Run generated prompts on various LLM platforms:

```bash
python 02_run_prompt/02_run_prompt_recurrent.py \
  -d <input_dir> \
  -i <prompt_file> \
  -o <output_dir> \
  -l <model_name>
```

### 4. Fine-tuning (Optional)

Fine-tune models on antibody sequence data:

**GPT Fine-tuning:**
```bash
python finetune/gpt_create_finetune.py
```

**Gemini Fine-tuning:**
```bash
python finetune/gemini_create_finetune.py
```

## Example Usage

### End-to-End Pipeline

1. **Select demonstrations:**
```bash
python 00_demonstration_selection/00_demonstration.py \
  -d ./data \
  -i scenario1_naiveness_mouse \
  -o ./demonstrations \
  -m naiveness_mouse
```

2. **Generate prompts:**
```bash
python 01_prompt_generation/scenario1_prompt_fewshot_ab_naiveness.py \
  -d ./demonstrations \
  -i scenario1_naiveness_mouse_similar_32_demonstrations.json \
  -t ./data/scenario1_naiveness_mouse \
  -o ./prompts \
  -m naiveness_mouse
```

3. **Run inference:**
```bash
python 02_run_prompt/02_run_prompt_recurrent.py \
  -d ./prompts \
  -i scenario1_naiveness_mouse_similar_32 \
  -o ./results \
  -l gpt-4o-mini-2024-07-18 \
  -a chat.completion \
  -t 0.2
```
