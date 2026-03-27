# Iterative Agentic Framework for Code Summarization
This repository is the replication package for the paper *From Draft to Precision: Iterative Agentic Framework for Intent-Aware Code Summarization*. It implements an iterative generator–evaluator–assessor–reviser loop, enhanced with external tools for content/context extraction and classifier-voting example selection.

---

## 1 Project Summary

The framework progressively refines code summaries to better align with developer intent. A run consists of the following stages:

1. **Generate** an initial one-sentence summary from the code and the requested intent.
2. **Evaluate** the summary on three dimensions: `intent_alignment`, `content_adequacy`, and `usefulness`.
3. **Plan** up to three concise revision actions.
4. **Revise** the previous summary into a new one-sentence summary.
5. **Stop** when the average evaluator score reaches the threshold, or when the maximum number of revision rounds is reached.

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
We use an intent-annotated subset of the [CodeSearchNet-Java](https://github.com/microsoft/CodeXGLUE) dataset. It contains code–comment pairs annotated with What, Why, How-it-is-done, and Property. Comments labeled as “Others” and rare categories like “How-to-use” are excluded. Scripts to preprocess and build this dataset are in `src/data/`. Classifier training for intent labeling is in `src/voting_classifier/`.

### 2.3 Tools
Two key tools support the framework, defined in `tool_module.py`:
- `get_context`: calls a Java parser JAR (configured via `JAVA_PARSER_JAR`) to extract content information such as docstrings, targets, callees, and callers.
- `get_examples`: retrieves candidate examples using `construction.instance_selection` and filters them with a finetuned classifier (local or HTTP service), applying majority or weighted voting.

### 2.4 Classifier
The script `prediction.py` performs voting across multiple classifier checkpoints:
```python
python prediction.py \
  --input ./data/test.jsonl \
  --output ./output/preds.jsonl \
  --checkpoints ckpt_a.pt,ckpt_b.pt,ckpt_c.pt \
  --vote majority
```
It supports majority and weighted voting, both locally and via HTTP endpoints. Metrics (accuracy, macro precision/recall/F1) are saved alongside predictions.

### 2.5 Agent Framework
Run the iterative agentic summarization with:
```python
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

[//]: # (## 3 Results)

[//]: # (Outputs include per-intent BLEU, ROUGE-L, and METEOR scores, with detailed JSONL logs and metrics.)

---

## 3. Prompt Templates

This section summarizes the prompt logic currently implemented in `multi_agent.py`.

### 3.1 Intent-specific shorthand prompts

The script defines the following intent-specific one-sentence prompt seeds:

```python
CLS_PROMPT = {
    'what': 'Please generate a short comment in one sentence describing what this function does and its primary purpose:',
    'property': 'Please generate a short comment in one sentence highlighting a key property of this function:',
    'done': 'Please generate a short comment in one sentence explaining how this function works and what it does internally:',
    'why': 'Please generate a short comment in one sentence explaining why this function work:'
}
```

These strings are useful for reproducing intent-specific prompting or one-shot baselines.

### 3.2 Generator prompt used in the current loop

**System prompt**

```text
You write precise one-sentence code comments aligned with a requested intent.
```

**User prompt template**

```text
Intent: <IntentName>
Code:
<code>
Return only one sentence.
```

**Example**

```text
[system]
You write precise one-sentence code comments aligned with a requested intent.

[user]
Intent: What
Code:
public int add(int a, int b) { return a + b; }
Return only one sentence.
```

Possible output:

```text
Returns the sum of the two input integers.
```

### 3.3 Assessor prompt

**System prompt**

```text
You are an evaluator. Output a JSON with numeric fields intent_alignment, content_adequacy, usefulness scored from 1 to 5.
```

**User payload format**

```json
{
  "intent": "What",
  "code": "public int add(int a, int b) { return a + b; }",
  "summary": "Returns the sum of the two input integers."
}
```

Possible output:

```json
{
  "intent_alignment": 5,
  "content_adequacy": 4,
  "usefulness": 4
}
```

### 3.4 Planner prompt

**System prompt**

```text
You are a planner. Given intent, code, current summary and scores, propose up to 3 concise revision plans. Output JSON {"plans": [..]}
```

**User payload format**

```json
{
  "intent": "What",
  "scores": {
    "intent_alignment": 4,
    "content_adequacy": 3,
    "usefulness": 4
  },
  "code": "public int clamp(int x, int low, int high) { return Math.max(low, Math.min(x, high)); }",
  "summary": "Adjusts the value."
}
```

Possible output:

```json
{
  "plans": [
    "Clarify that the method constrains the input to a lower and upper bound.",
    "Mention the range semantics rather than saying only that it adjusts the value.",
    "Use concise wording and keep the sentence focused on the What intent."
  ]
}
```

### 3.5 Reviser prompt

**System prompt**

```text
You revise code comments. Rewrite into one sentence aligned with the intent, following the plans and using supply info when helpful. Return only the revised sentence.
```

**User payload format**

```json
{
  "intent": "What",
  "code": "public int clamp(int x, int low, int high) { return Math.max(low, Math.min(x, high)); }",
  "previous_summary": "Adjusts the value.",
  "plans": [
    "Clarify that the method constrains the input to a lower and upper bound.",
    "Mention the range semantics rather than saying only that it adjusts the value."
  ],
  "supply_info": "Example 1:\nCode:\npublic int normalize(int x, int low, int high) { ... }\nComment:\nConstrains the input value to the target range."
}
```

Possible output:

```text
Constrains the input value to the specified lower and upper bounds.
```
