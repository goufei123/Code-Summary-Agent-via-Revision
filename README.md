# Code-Summary-Agent-via-Revision
This repository is the replication package for the paper *From Draft to Precision: Iterative Agentic Framework for Intent-Aware Code Summarization*. It implements an iterative generator–evaluator–planner–reviser loop, enhanced with external tools for content/context extraction and classifier-voting example selection.

## 1 Project Summary
The framework progressively refines code summaries to align with developer intents (What, Why, How-it-is-done, Property). It integrates:
- A generator for initial summaries.
- An evaluator that scores intent alignment, content adequacy, and usefulness.
- A planner that proposes revision strategies.
- A reviser that incorporates revision plans and supply information.
Supply information is built from **content-aware analysis** (via a Java parser tool) and **example-aware retrieval** (via instance selection plus classifier voting). Classifier-voting ensembles multiple finetuned classifiers to filter high-confidence examples. Experiments show significant improvements over baseline summarization models.

## 2 Get Started
### 2.1 Requirements
* OS: Ubuntu 20.04 or later
* Python 3.10+
* PyTorch (compatible with CUDA 11.8/12.x)
* Hugging Face Transformers (latest stable version)
* datasets
* numpy
* tqdm
* jsonlines
* Java Runtime (for running the parser JAR)

### 2.2 Dataset
We use an intent-annotated subset of the [CodeSearchNet-Java](https://github.com/microsoft/CodeXGLUE) dataset. It contains code–comment pairs annotated with What, Why, How-it-is-done, and Property. Comments labeled as “Others” and rare categories like “How-to-use” are excluded. Scripts to preprocess and build this dataset are in `src/data/`. Classifier training for intent labeling is in `src/voting_classifier/`.

### 2.3 Tools
Two key tools support the framework, defined in `tool_module.py`:
- `get_context`: calls a Java parser JAR (configured via `JAVA_PARSER_JAR`) to extract content information such as docstrings, targets, callees, and callers.
- `get_examples`: retrieves candidate examples using `construction.instance_selection` and filters them with a finetuned classifier (local or HTTP service), applying majority or weighted voting.

### 2.4 Classifier Voting
The script `prediction.py` performs voting across multiple classifier checkpoints:
```
python prediction.py \
  --input ./data/test.jsonl \
  --output ./output/preds.jsonl \
  --checkpoints ckpt_a.pt,ckpt_b.pt,ckpt_c.pt \
  --vote majority
```
It supports majority and weighted voting, both locally and via HTTP endpoints. Metrics (accuracy, macro precision/recall/F1) are saved alongside predictions.

### 2.5 Agent Framework
Run the iterative agentic summarization with:
```
python multi_agent.py \
  --model deepseek \
  --prompt_filename ./output/cls_examples_test_all.jsonl \
  --output_dir ./output/eval_result/ \
  --max_rounds 3 \
  --threshold 4.0 \
  --temperature 0.2
```
This loop generates, evaluates, plans, and revises summaries until a quality threshold or max rounds is reached. Supply information integrates outputs from `get_context` and `get_examples`.

### 2.6 Utils
General helper functions are in `utils.py`:
- `set_seed(seed)`: set random seeds for reproducibility.
- `read_jsonl(path)`, `write_jsonl(path, rows)`: handle JSONL files.
- `normalize_text(s)`: normalize text to lowercase and strip extra spaces.
- `average_dicts(dicts)`: average numeric values across dictionaries.

## 3 Results
Outputs include per-intent BLEU, ROUGE-L, and METEOR scores, with detailed JSONL logs and metrics.