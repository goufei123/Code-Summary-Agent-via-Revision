# Iterative Agentic Framework for Code Summarization
This repository is the replication package for the paper *From Draft to Precision: Iterative Agentic Framework for Intent-Aware Code Summarization*. It implements the paper's Generator--Reviewer loop: the Summarizer drafts and rewrites summaries, while the Reviewer contains an Assessor and a Planner that score drafts and produce revision plans. The loop is enhanced with support modules for content/context extraction and classifier-voting example selection.

---

## 1 Project Summary

The framework progressively refines code summaries to better align with developer intent. A run consists of the following stages:

1. **Summarize** an initial one-sentence summary from the code and the requested intent.
2. **Assess** the summary on three dimensions: `intent_alignment`, `content_adequacy`, and `usefulness`.
3. **Plan** up to three concise revision actions when the draft does not pass the threshold.
4. **Revise** the previous summary with the revision plans and optional support-module evidence.
5. **Stop** when the average Assessor score reaches the threshold, or when the maximum number of revision rounds is reached.

The implementation uses a LangGraph state machine to realize this loop.

---

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

```bash
pip install openai langgraph jsonlines tqdm numpy datasets transformers torch
```

#### API keys
`multi_agent.py` reads model credentials from environment variables:

- `OPENAI_API_KEY` for `--model gpt`
- `DEEPSEEK_API_KEY` for `--model deepseek`


### 2.2 Dataset
We use an intent-annotated subset of the [CodeSearchNet-Java](https://github.com/microsoft/CodeXGLUE) dataset. It contains code-comment pairs annotated with What, Why, How-it-is-done, and Property. Comments labeled as “Others” and the sparse “How-to-use” category are excluded. The resulting evaluation set contains 10,810 samples across the four evaluated intents. `src/agent_framework/dataloader.py` provides the conversion/filtering utility for JSONL inputs. Voting-based classifier inference for intent labeling is in `src/voting_classifier/`.

### 2.3 Tools
Two key support modules are defined in `src/agent_framework/tool_module.py`:
- `get_context`: calls a Java parser JAR (configured via `JAVA_PARSER_JAR`) to extract content information such as docstrings, targets, callees, and callers.
- `get_examples`: retrieves candidate examples using `construction.instance_selection` and filters them with a finetuned classifier (local or HTTP service), applying majority or weighted voting.

### 2.4 Classifier
The script `src/voting_classifier/prediction.py` performs voting across multiple classifier checkpoints:
```bash
python src/voting_classifier/prediction.py \
  --input ./data/test.jsonl \
  --output ./output/preds.jsonl \
  --checkpoints ckpt_a.pt,ckpt_b.pt,ckpt_c.pt \
  --vote majority
```
It supports majority and weighted voting, both locally and via HTTP endpoints. Metrics (accuracy, macro precision/recall/F1) are saved alongside predictions.

### 2.5 Agent Framework
Run the iterative agentic summarization with:
```bash
python src/agent_framework/multi_agent.py \
  --summarizer_model gpt \
  --reviewer_model gpt \
  --prompt_filename ./data/cls_examples_test_all.jsonl \
  --output_dir ./output/eval_result/ \
  --max_rounds 3 \
  --threshold 4.0 \
  --temperature 0.5 \
  --top_p 0.75
```
This loop summarizes, assesses, plans, and revises until the Assessor score reaches the threshold or the maximum number of revision rounds is reached. The default parameters match the revised manuscript setting: `max_rounds=3`, `threshold=4.0`, `temperature=0.5`, and `top_p=0.75`. Unless otherwise specified, the paper uses GPT-4o as the Reviewer backbone; `--reviewer_model` is exposed to run Reviewer-backbone sensitivity checks.

For backward compatibility, `--model` remains as a deprecated alias for `--summarizer_model`.

### 2.6 Utils
General helper functions are in `utils.py`:
- `set_seed(seed)`: set random seeds for reproducibility.
- `read_jsonl(path)`, `write_jsonl(path, rows)`: handle JSONL files.
- `normalize_text(s)`: normalize text to lowercase and strip extra spaces.
- `average_dicts(dicts)`: average numeric values across dictionaries.

[//]: # (## 3 Results)

[//]: # (Outputs include per-intent BLEU, ROUGE-L, and METEOR scores, with detailed JSONL logs and metrics.)

---

## 3. Additional Documentation

- Prompt templates and example payloads are documented in [PROMPTS.md](PROMPTS.md).
