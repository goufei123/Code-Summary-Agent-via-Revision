import subprocess
import json
import os
import re
import urllib.request
import urllib.error
from construction import instance_selection

def _run_java_parser(code):
    jar_path = os.getenv("JAVA_PARSER_JAR", "./parser.jar")
    if not code:
        return {}
    try:
        p = subprocess.Popen(["java", "-jar", jar_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out, err = p.communicate(code, timeout=float(os.getenv("JAVA_PARSER_TIMEOUT", "30")))
        if p.returncode != 0:
            return {}
        s = out.strip()
        try:
            return json.loads(s)
        except Exception:
            for line in s.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    return json.loads(line)
                except Exception:
                    pass
            return {}
    except Exception:
        return {}

def _extract_doc(doc):
    if not isinstance(doc, str) or not doc:
        return ""
    lines = doc.splitlines()
    allowed = ("@param", "@return", "@throws", "@since", "@deprecated")
    buf = []
    cur = None
    for line in lines:
        t = line.strip()
        if any(t.startswith(tag) for tag in allowed):
            cur = t
            buf.append(cur)
        elif cur and (line.startswith(" ") or line.startswith("\t")):
            buf[-1] = buf[-1] + " " + t
        else:
            cur = None
    x = []
    for a in buf:
        x.append(re.sub(r"\{@\w+\s+([^}]+)\}", r"\1", a))
    return "\n".join(x)

def get_content_info(code):
    j = _run_java_parser(code)
    if not isinstance(j, dict):
        return {}
    out = {}
    if "comments" in j:
        out["comments"] = j.get("comments")
    if "target" in j:
        out["target"] = j.get("target")
    if "callees" in j:
        out["callees"] = j.get("callees")
    if "callers" in j:
        out["callers"] = j.get("callers")
    return out

def get_context(parsed_results_or_obj):
    code = parsed_results_or_obj.get("code", "") if isinstance(parsed_results_or_obj, dict) else ""
    info = get_content_info(code)
    parts = []
    doc = _extract_doc(info.get("comments", ""))
    if doc:
        parts.append("Docstring:\n" + doc)
    tgt = info.get("target")
    if isinstance(tgt, str) and tgt:
        parts.append("Location: " + tgt)
    cals = info.get("callees", [])
    if isinstance(cals, list) and cals:
        names = []
        for i in range(0, len(cals), 3):
            g = cals[i:i+3]
            if len(g) >= 2 and isinstance(g[1], str) and g[1]:
                names.append(g[1])
        if names:
            parts.append("Calls: " + ", ".join(names))
    clrs = info.get("callers", [])
    if isinstance(clrs, list) and clrs:
        names = []
        for i in range(0, len(clrs), 3):
            g = clrs[i:i+3]
            if len(g) >= 3 and isinstance(g[2], str) and g[2]:
                names.append(g[2])
        if names:
            parts.append("Used by: " + ", ".join(names))
    if parts:
        parts.append("Based on the above information,")
    return "\n\n".join(parts)

def _predict_intents_http(items):
    url = os.getenv("CLASSIFIER_ENDPOINT", "")
    if not url:
        return []
    try:
        data = json.dumps({"items": items}).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=float(os.getenv("CLASSIFIER_TIMEOUT", "10"))) as resp:
            s = resp.read().decode("utf-8")
            j = json.loads(s)
            return j if isinstance(j, list) else j.get("items", [])
    except Exception:
        return []

def _predict_intents_py(items):
    try:
        from intent_classifier.api import predict_batch
        return predict_batch(items)
    except Exception:
        try:
            from intent_classifier.api import predict_intent
            out = []
            for it in items:
                r = predict_intent(it.get("code", ""), it.get("comment", ""))
                out.append(r)
            return out
        except Exception:
            return []

def _predict_intents(items):
    res = _predict_intents_http(items)
    if res:
        return res
    return _predict_intents_py(items)

def get_examples(intent, code):
    k_pool = int(os.getenv("EXAMPLE_POOL", "16"))
    k_return = int(os.getenv("EXAMPLE_TOPK", "3"))
    thr = float(os.getenv("EXAMPLE_MIN_CONF", "0.0"))
    try:
        try:
            pool = instance_selection(intent, code, k_pool)
        except TypeError:
            pool = instance_selection(intent, code)
        items = []
        for e in pool or []:
            if isinstance(e, dict):
                c = e.get("code", "")
                cm = e.get("comment", "")
                if c and cm:
                    items.append({"code": c, "comment": cm})
            elif isinstance(e, (list, tuple)) and len(e) >= 2:
                c = e[0]
                cm = e[1]
                if isinstance(c, str) and isinstance(cm, str):
                    items.append({"codimport os
import json
import jsonlines
import random
import numpy as np
import torch

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def read_jsonl(path: str):
    with jsonlines.open(path, 'r') as reader:
        return [obj for obj in reader]

def write_jsonl(path: str, rows: list):
    with jsonlines.open(path, 'w') as writer:
        writer.write_all(rows)

def normalize_text(s: str):
    import re
    s = s.lower().strip()
    s = re.sub(r'\s+', ' ', s)
    return s

def average_dicts(dicts: list):
    if not dicts:
        return {}
    keys = dicts[0].keys()
    avg_dict = {}
    for key in keys:
        values = [d[key] for d in dicts if key in d]
        avg_dict[key] = sum(values) / len(values) if values else 0
    return avg_dict
e": c, "comment": cm})
        if not items:
            return []
        preds = _predict_intents(items)
        scored = []
        for it, pr in zip(items, preds if preds else [None]*len(items)):
            if isinstance(pr, dict):
                lab = str(pr.get("intent", "")).lower()
                sc = float(pr.get("score", 0.0))
            elif isinstance(pr, (list, tuple)) and len(pr) >= 2:
                lab = str(pr[0]).lower()
                sc = float(pr[1])
            else:
                lab = ""
                sc = 0.0
            if lab == str(intent).lower() and sc >= thr:
                scored.append((sc, it))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [x[1] for x in scored[:k_return]]
    except Exception:
        return []