# Code-Summary-Agent-via-Revision
This repository is the replication package for the paper *From Draft to Precision: Iterative Agentic Framework for Intent-Aware Code Summarization*.

## Content
TBD

[//]: # (1. [Project Summary]&#40;#1-Project-Summary&#41;<br>)

[//]: # (2. [Get Started]&#40;#2-Get-Started&#41;<br>)

[//]: # (&ensp;&ensp;[2.1 Requirements]&#40;#21-Requirements&#41;<br>)

[//]: # (&ensp;&ensp;[2.2 Dataset]&#40;#22-Dataset&#41;<br>)

[//]: # (&ensp;&ensp;[2.3 Comment Intent Labeling Tool]&#40;#23-Comment-Intent-Labeling-Tool&#41;<br>)

[//]: # (&ensp;&ensp;[2.4 Developer-Intent Driven Comment Generator]&#40;#24-Developer-Intent-Driven-Comment-Generator&#41;<br>)

[//]: # (3. [Application]&#40;#3-Application&#41;<br>)

## 1 Project Summary

This paper proposes an iterative agentic framework for intent-aware code summarization, which leverages a generator–evaluator design to progressively refine code summaries based on developer intents. The framework integrates multiple specialized agents that collaboratively generate and evaluate comments to improve summary precision and relevance. Key contributions include an extensive empirical study on intent-aware summarization, comprehensive experiments demonstrating significant improvements over state-of-the-art baselines, and the introduction of a novel iterative approach that effectively captures developer intent in code comments.

## 2 Get Started
### 2.1 Requirements
* OS: Ubuntu 20.04 or later
* Python 3.10+
* PyTorch (compatible with CUDA 11.8/12.x)
* Hugging Face Transformers (latest stable version)
* datasets
* numpy
* tqdm

### 2.2 Dataset
#### Intent-annotated CSN-Java dataset
The benchmark dataset used in this work is an intent-annotated subset of the CodeSearchNet-Java [CSN-java](https://github.com/microsoft/CodeXGLUE) dataset. It contains code-comment pairs labeled with four main developer intent categories: What, Why, How-it-is-done, and Property. The dataset is constructed by following prior work on intent annotation, excluding comments labeled as "Others" and rare categories such as "How-to-use". Scripts to preprocess and generate this dataset are provided in the `src/data/` directory. The dataset download link and detailed instructions are available there.

### 2.3 Comment Intent Labeling Tool
Intent labels are generated using a classifier fine-tuned on human-annotated data. The fine-tuning and prediction scripts for the intent classifier are provided in the `src/intent_classifier/` directory.

### 2.4 Developer-Intent Driven Comment Generator
To run the iterative agentic framework for intent-aware code summarization:

1. Navigate to the `src/agent_framework/` directory.
2. Run the framework with the desired configuration file:
```
python run_agent_framework.py --config configs/{config_file}.yaml
```
3. Results and logs will be saved in the `outputs/` directory.