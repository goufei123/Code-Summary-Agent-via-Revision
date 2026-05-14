import json
import os
import re
import subprocess
import urllib.request
from typing import Any, Dict, List

try:
    from construction import instance_selection
except Exception:
    instance_selection = None


def _run_java_parser(code: str) -> Dict[str, Any]:
    jar_path = os.getenv("JAVA_PARSER_JAR", "./parser.jar")
    if not code:
        return {}
    try:
        process = subprocess.Popen(
            ["java", "-jar", jar_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, _ = process.communicate(code, timeout=float(os.getenv("JAVA_PARSER_TIMEOUT", "30")))
        if process.returncode != 0:
            return {}
        text = stdout.strip()
        try:
            return json.loads(text)
        except Exception:
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    return json.loads(line)
                except Exception:
                    continue
            return {}
    except Exception:
        return {}


def _extract_doc(doc: Any) -> str:
    if not isinstance(doc, str) or not doc:
        return ""
    allowed = ("@param", "@return", "@throws", "@since", "@deprecated")
    blocks = []
    current = None
    for line in doc.splitlines():
        text = line.strip()
        if any(text.startswith(tag) for tag in allowed):
            current = text
            blocks.append(current)
        elif current and (line.startswith(" ") or line.startswith("\t")):
            blocks[-1] = blocks[-1] + " " + text
        else:
            current = None
    return "\n".join(re.sub(r"\{@\w+\s+([^}]+)\}", r"\1", block) for block in blocks)


def _content_fields(parsed: Dict[str, Any]) -> Dict[str, Any]:
    return {key: parsed[key] for key in ("comments", "target", "callees", "callers") if key in parsed}


def get_content_info(code: str) -> Dict[str, Any]:
    parsed = _run_java_parser(code)
    return _content_fields(parsed) if isinstance(parsed, dict) else {}


def _coerce_content_info(parsed_results_or_obj: Any) -> Dict[str, Any]:
    if not isinstance(parsed_results_or_obj, dict):
        return {}
    if any(key in parsed_results_or_obj for key in ("comments", "target", "callees", "callers")):
        return _content_fields(parsed_results_or_obj)
    nested = parsed_results_or_obj.get("parsed_results")
    if isinstance(nested, dict) and any(key in nested for key in ("comments", "target", "callees", "callers")):
        return _content_fields(nested)
    return get_content_info(parsed_results_or_obj.get("code", ""))


def _extract_grouped_names(values: Any, name_index: int) -> List[str]:
    if not isinstance(values, list):
        return []
    names = []
    for i in range(0, len(values), 3):
        group = values[i:i + 3]
        if len(group) > name_index and isinstance(group[name_index], str) and group[name_index]:
            names.append(group[name_index])
    return names


def get_context(parsed_results_or_obj: Any) -> str:
    info = _coerce_content_info(parsed_results_or_obj)
    parts = []

    doc = _extract_doc(info.get("comments", ""))
    if doc:
        parts.append("Docstring:\n" + doc)

    target = info.get("target")
    if isinstance(target, str) and target:
        parts.append("Location: " + target)

    callees = _extract_grouped_names(info.get("callees", []), 1)
    if callees:
        parts.append("Calls: " + ", ".join(callees))

    callers = _extract_grouped_names(info.get("callers", []), 2)
    if callers:
        parts.append("Used by: " + ", ".join(callers))

    if parts:
        parts.append("Based on the above information,")
    return "\n\n".join(parts)


def _predict_intents_http(items: List[Dict[str, str]]) -> List[Any]:
    url = os.getenv("CLASSIFIER_ENDPOINT", "")
    if not url:
        return []
    try:
        data = json.dumps({"items": items}).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=float(os.getenv("CLASSIFIER_TIMEOUT", "10"))) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return payload if isinstance(payload, list) else payload.get("items", [])
    except Exception:
        return []


def _predict_intents_py(items: List[Dict[str, str]]) -> List[Any]:
    try:
        from intent_classifier.api import predict_batch
        return predict_batch(items)
    except Exception:
        try:
            from intent_classifier.api import predict_intent
            return [predict_intent(item.get("code", ""), item.get("comment", "")) for item in items]
        except Exception:
            return []


def _predict_intents(items: List[Dict[str, str]]) -> List[Any]:
    predictions = _predict_intents_http(items)
    if predictions:
        return predictions
    return _predict_intents_py(items)


def _normalize_intent(label: Any) -> str:
    text = str(label).lower().strip().replace("_", "-")
    if text in {"how-it-is-done", "how it is done", "how"}:
        return "done"
    return text


def _candidate_items(pool: List[Any]) -> List[Dict[str, str]]:
    items = []
    for example in pool or []:
        if isinstance(example, dict):
            code = example.get("code", "")
            comment = example.get("comment", "")
        elif isinstance(example, (list, tuple)) and len(example) >= 2:
            code, comment = example[0], example[1]
        else:
            continue
        if isinstance(code, str) and isinstance(comment, str) and code and comment:
            items.append({"code": code, "comment": comment})
    return items


def get_examples(intent: str, code: str) -> List[Dict[str, str]]:
    if instance_selection is None:
        return []

    k_pool = int(os.getenv("EXAMPLE_POOL", "16"))
    k_return = int(os.getenv("EXAMPLE_TOPK", "3"))
    min_confidence = float(os.getenv("EXAMPLE_MIN_CONF", "0.0"))
    target_intent = _normalize_intent(intent)

    try:
        try:
            pool = instance_selection(target_intent, code, k_pool)
        except TypeError:
            pool = instance_selection(target_intent, code)

        items = _candidate_items(pool)
        if not items:
            return []

        predictions = _predict_intents(items)
        scored = []
        for item, prediction in zip(items, predictions if predictions else [None] * len(items)):
            if isinstance(prediction, dict):
                label = _normalize_intent(prediction.get("intent", ""))
                score = float(prediction.get("score", 0.0))
            elif isinstance(prediction, (list, tuple)) and len(prediction) >= 2:
                label = _normalize_intent(prediction[0])
                score = float(prediction[1])
            else:
                label = ""
                score = 0.0
            if label == target_intent and score >= min_confidence:
                scored.append((score, item))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in scored[:k_return]]
    except Exception:
        return []
