# Demonstration Selection

This module handles the selection of few-shot demonstrations (examples) for in-context learning. It implements various sampling strategies to select the most effective examples from the training set for each test instance.

## Overview
<p align="center">
  <img src="https://github.com/user-attachments/assets/967e6691-49f0-441e-bb56-ae15d3f1aea9" width="100%">
</p>

Few-shot learning performance heavily depends on the quality and relevance of demonstration examples. This module provides multiple strategies to select demonstrations, ranging from random sampling to sophisticated similarity-based approaches.

## Input Data Requirements

### CSV Format
Training and test sets must be CSV files with:
- A sequence column (name specified by `mode`)
- A `label` column (0/1 for binary, 0-N for multi-class)

**Example:**
```csv
seq,label
EVQLVESGGGLVQPGGSLRLSCAASG...,1
QVQLQQSGAELARPGASVKMSCKAS...,0
```

## Scripts

### 1. `00_demonstration.py` - Sequence-based Selection

Selects demonstrations based on sequence similarity using Levenshtein distance.

**Usage:**
```bash
python 00_demonstration.py \
  -d <input_dir> \
  -i <input_file> \
  -o <output_dir> \
  -m <mode>
```

**Arguments:**
- `-d, --input_dir`: Directory containing input data
- `-i, --input`: Input dataframe name (without `_train_set`/`_test_set` suffix)
- `-o, --output_dir`: Output directory for demonstration files
- `-m, --mode`: Sequence mode
  - `H`: Heavy chain only
  - `HL`: Heavy and light chain
  - `naiveness_mouse`: Mouse naiveness prediction
  - `naiveness_rhesus`: Rhesus macaque naiveness prediction

**Sampling Methods:**
1. **random**: Random selection of balanced positive/negative examples
2. **similar**: Most similar examples based on Levenshtein distance
3. **similar_reverse**: Least similar examples (diversity-based)
4. **similar_balanced_posneg**: Similar examples, balanced classes (positive then negative)
5. **similar_balanced_pos**: Only similar positive examples
6. **similar_balanced_negpos**: Similar examples, balanced classes (negative then positive)
7. **similar_balanced_random**: Similar examples, balanced classes, randomly shuffled

**Sample Sizes:** 2, 4, 8, 16, 32, 50, 100, 200

### 2. `00_demonstration_emb.py` - Embedding-based Selection

Selects demonstrations using pre-computed protein language model (pLM) embeddings and cosine similarity.

**Usage:**
```bash
python 00_demonstration_emb.py \
  -d <input_dir> \
  -i <input_file> \
  -p <plm_name> \
  -o <output_dir> \
  -m <mode>
```

### 3. `00_demonstration_multi_class.py` - Multi-class Selection

Extends demonstration selection to multi-class classification tasks (e.g., isotype classification).

**Usage:**
```bash
python 00_demonstration_multi_class.py \
  -d <input_dir> \
  -i <input_file> \
  -o <output_dir> \
  -m <mode> \
  -n <num_classes>
```

### Embedding Format
For embedding-based selection:
- PyTorch `.pt` files containing embedding tensors
- NumPy `.npy` files containing labels
- Separate files for positive/negative classes

## Example Workflow

```bash
# 1. Select demonstrations using sequence similarity
python 00_demonstration.py \
  -d ./data \
  -i scenario1_naiveness_mouse \
  -o ./demonstrations \
  -m naiveness_mouse

# 2. Select demonstrations using embeddings (if available)
python 00_demonstration_emb.py \
  -d ./data/embeddings \
  -i scenario1_naiveness_mouse \
  -p antiberty \
  -o ./demonstrations \
  -m naiveness_mouse

# 3. For multi-class tasks (e.g., isotype classification)
python 00_demonstration_multi_class.py \
  -d ./data \
  -i scenario3_isotype \
  -o ./demonstrations \
  -m H \
  -n 4
```

These demonstration files are then used as input for the prompt generation step (see `01_prompt_generation/`).
