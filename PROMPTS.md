# Prompt Templates

This file summarizes the prompt logic currently implemented in `src/agent_framework/multi_agent.py`.

## Intent-Specific Shorthand Prompts

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

## Summarizer Prompt Used in the Current Loop

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

## Reviewer: Assessor Prompt

**System prompt**

```text
You are the Assessor in a Reviewer agent. Output a JSON with numeric fields intent_alignment, content_adequacy, usefulness scored from 1 to 5.
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

## Reviewer: Planner Prompt

**System prompt**

```text
You are the Planner in a Reviewer agent. Given intent, code, current summary and scores, propose up to 3 concise revision plans. Output JSON {"plans": [..]}
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

## Summarizer Revision Prompt

**System prompt**

```text
You are the Summarizer. Revise the code comment into one sentence aligned with the intent, following the plans and using supply info when helpful. Return only the revised sentence.
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
