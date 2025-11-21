# Prompt Generation

This module converts demonstration files and test data into complete prompts ready for LLM inference. It implements three classification scenarios with support for zero-shot, few-shot, and Chain-of-Thought (CoT) prompting strategies.

## Overview
<p align="center">
  <img src="https://github.com/user-attachments/assets/2a4b3ae8-7072-4f04-bc3b-ae0a127cbecb" width="100%">
</p>
Each script generates task-specific prompts that include:
- System instructions defining the task
- Few-shot examples (for few-shot prompts)
- Chain-of-thought reasoning steps (for CoT prompts)
- Test instance to classify
- Output format requirements

## Classification Scenarios

### Scenario 1: Antibody Humanness/Naiveness Prediction

Predicts whether an antibody sequence is human or from another organism (Mouse/Rhesus macaque).

#### Files

**Few-shot:**
- `scenario1_prompt_fewshot_ab_naiveness.py` - Sequence-based few-shot
- `scenario1_prompt_fewshot_ab_naiveness_emb.py` - Embedding-based few-shot

**Chain-of-Thought:**
- `scenario1_cot_prompt_fewshot_rhesus.py` - CoT reasoning for rhesus macaque

**Zero-shot:**
- `scenario1_prompt_zeroshot_ab_naiveness.py` - No demonstrations

#### Usage

**Few-shot:**
```bash
python scenario1_prompt_fewshot_ab_naiveness.py \
  -d <demonstration_dir> \
  -i <demonstration_file> \
  -t <test_file> \
  -o <output_dir> \
  -m <mode>
```

**Zero-shot:**
```bash
python scenario1_prompt_zeroshot_ab_naiveness.py \
  -t <test_file> \
  -o <output_dir> \
  -n <output_name> \
  -m <mode>
```

**Arguments:**
- `-d, --input_dir`: Directory containing demonstration files
- `-i, --input`: Demonstration file name (must end with `_demonstrations.json`)
- `-t, --test`: Test set path (without `_test_set` suffix)
- `-o, --output_dir`: Output directory for prompts
- `-m, --mode`: Sequence mode (`naiveness_mouse`, `naiveness_rhesus`, `H`, `HL`)
- `-n, --output_name`: Custom output filename (zero-shot only)

### Scenario 2: SARS-CoV-2 Specificity Prediction

Predicts whether an antibody binds to SARS-CoV-2 spike protein.

#### Files

**Few-shot:**
- `scenario2_prompt_fewshot_sarscov2_specificity.py`

**Chain-of-Thought:**
- `scenario2_cot_prompt_fewshot_sarscov2_specificity.py`
- `scenario2_cot_manual_prompt_fewshot_sarscov2_specificity.py` - Manually crafted CoT

**Zero-shot:**
- `scenario2_prompt_zeroshot_sarscov2_specificity.py`

#### Usage

```bash
python scenario2_prompt_fewshot_sarscov2_specificity.py \
  -d <demonstration_dir> \
  -i <demonstration_file> \
  -t <test_file> \
  -o <output_dir> \
  -m <mode>
```

### Scenario 3: Antibody Isotype Classification

Multi-class classification of antibody isotype (IGHA, IGHD, IGHG, IGHM) or light chain type (IGLC, IGKC).

#### Files

**Few-shot:**
- `scenario3_prompt_fewshot_isotype.py`

**Chain-of-Thought:**
- `scenario3_cot_prompt_fewshot_isotype.py`

**Zero-shot:**
- `scenario3_prompt_zeroshot_isotype.py`

#### Usage

```bash
python scenario3_prompt_fewshot_isotype.py \
  -d <demonstration_dir> \
  -i <demonstration_file> \
  -t <test_file> \
  -o <output_dir> \
  -m <mode>
```

**Mode options:**
- `heavy_seq`: Heavy chain isotype (4 classes: 0=IGHA, 1=IGHD, 2=IGHG, 3=IGHM)
- `light_seq`: Light chain type (2 classes: 0=IGLC, 1=IGKC)

## Prompting Strategies

### 1. Zero-shot Prompting

Direct classification without examples, relying solely on the model's pre-trained knowledge.

### 2. Few-shot Prompting

Provides N demonstration examples to guide the model's predictions.

**Variants:**
- **No tag**: Examples listed directly in sequence
- **Tag**: Examples wrapped in `<example>` tags for better separation

### 3. Chain-of-Thought (CoT) Prompting

Encourages the model to explain its reasoning process before making predictions.

## Output Formats

### Few-shot Prompts
Generates two variants (where applicable):

1. **No tag version:** `{output_name}_no_tag_prompt.json`
2. **Tag version:** `{output_name}_tag_prompt.json`

### Zero-shot Prompts
Single output: `{output_name}_zero_shot_prompt.json`

### CoT Prompts
Single output: `cot_{output_name}_no_tag_prompt.json`

### JSON Structure
```json
[
  "Prompt text for test instance 1...",
  "Prompt text for test instance 2...",
  ...
]
```

## Example Prompts

### Scenario 1: Humanness (Few-shot)
```
You are an AI model specializing in immunology. Your task is to predict
the humanness of antibody using only the amino acid sequences provided...

Antibody: EVQLVESGGGLVQPGGSLRLSCAASG...
Humanness: 1

Antibody: QVQLQQSGAELARPGASVKMSCKAS...
Humanness: 0

Now, predict the humanness of antibody for the following sequence:
Antibody: QVQLVQSGAEVKKPGASVKVSCKAS...
Humanness:
```

### Scenario 2: SARS-CoV-2 Specificity (CoT)
```
You are an AI model specializing in immunology. Your task is to predict
the binding specificity of antibody to the SARS-CoV-2 spike protein...

[Few-shot examples]

Let's solve this problem by splitting it into steps.
Step 1: Analyze CDR regions for known SARS-CoV-2 binding motifs
Step 2: Consider sequence homology to validated binders

Now, predict the receptor binding specificity:
Antibody: QVQLVQSGAEVKKPGASVKVSCKAS...
Binding:
```

### Scenario 3: Isotype Classification (Multi-class)
```
You are an AI model specializing in immunology. Your task is to predict
the heavy chain isotype using only the amino acid sequences provided...

Given the amino acid sequences of an antibody, predict the heavy chain
isotype: IGHA (0), IGHD (1), IGHG (2), IGHM (3).

Please answer with *only* 0, 1, 2 or 3 for the last example.

[Few-shot examples]

Antibody: ASTKGPSVFPLAPSSKSTSGGTAALGCLVK...
Heavy chain isotype:
```

## Workflow Example

### Complete Pipeline for Scenario 1

```bash
# 1. Generate few-shot prompts (32 demonstrations, similar_balanced_random)
python scenario1_prompt_fewshot_ab_naiveness.py \
  -d ../demonstrations \
  -i fewshot_naiveness_mouse_32_similar_balanced_random_demonstrations.json \
  -t ../data/scenario1_naiveness_mouse \
  -o ../prompts \
  -m naiveness_mouse

# Output:
#   fewshot_naiveness_mouse_32_similar_balanced_random_no_tag_prompt.json
#   fewshot_naiveness_mouse_32_similar_balanced_random_tag_prompt.json

# 2. Generate CoT prompts
python scenario1_cot_prompt_fewshot_rhesus.py \
  -d ../demonstrations \
  -i fewshot_naiveness_rhesus_32_similar_balanced_random_demonstrations.json \
  -t ../data/scenario1_naiveness_rhesus \
  -o ../prompts \
  -m naiveness_rhesus

# Output:
#   cot_fewshot_naiveness_rhesus_32_similar_balanced_random_no_tag_prompt.json

# 3. Generate zero-shot baseline
python scenario1_prompt_zeroshot_ab_naiveness.py \
  -t ../data/scenario1_naiveness_mouse \
  -o ../prompts \
  -n scenario1_naiveness_mouse \
  -m naiveness_mouse

# Output:
#   scenario1_naiveness_mouse_zero_shot_prompt.json
```

## Input Requirements

### Demonstration Files
JSON files from `00_demonstration_selection/`:
```json
[
  [["SEQUENCE1", 1], ["SEQUENCE2", 0], ...],  # Demos for test instance 1
  [["SEQUENCE_A", 1], ["SEQUENCE_B", 1], ...], # Demos for test instance 2
  ...
]
```

### Test Files
CSV files with columns:
- Sequence column (specified by `mode`)
- `label` column

**Example:**
```csv
seq,label
EVQLVESGGGLVQPGGSLRLSCAASG...,1
QVQLQQSGAELARPGASVKMSCKAS...,0
```

## Best Practices

1. **Tag vs No-tag**: Test both variants; some models perform better with explicit tags
2. **CoT for complex tasks**: Use Chain-of-Thought for multi-step reasoning tasks
3. **Zero-shot baseline**: Always generate zero-shot prompts for comparison
4. **Prompt consistency**: Keep system instructions consistent across experiments
5. **Output format**: Clearly specify expected output format (e.g., "only 1 or 0")

## Customization

To create prompts for new scenarios:

1. Copy an existing scenario file
2. Modify the `create_prompt()` function:
   - Update system instructions
   - Change task description
   - Adjust label format
   - Add/remove CoT steps
3. Update output file naming conventions

**Example:**
```python
def create_prompt(test_example, fewshot_examples):
    prompt = "You are an AI model specializing in [DOMAIN]. "
    prompt += "Your task is to [TASK DESCRIPTION]...\n"

    # Add few-shot examples
    for example in fewshot_examples:
        prompt += f"Input: {example[0]}\nOutput: {example[-1]}\n"

    # Add test instance
    prompt += f"Input: {test_example}\nOutput:\n"
    return prompt
```

## Output

Generated prompt files are saved in the specified output directory and can be directly used for model inference in `02_run_prompt/`.

## Next Steps

After generating prompts, proceed to `02_run_prompt/` to run inference using your preferred LLM platform.
