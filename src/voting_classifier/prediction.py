import argparse
import os
import json
import jsonlines
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple
import urllib.request


def read_items(path: str, code_field: str, comment_field: str) -> List[Dict[str, Any]]:
    items = []
    with jsonlines.open(path) as reader:
        for obj in reader:
            code = obj.get(code_field, "")
            comment = obj.get(comment_field, "")
            items.append({"code": code, "comment": comment, "_raw": obj})
    return items


def write_items(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with jsonlines.open(path, "w") as w:
        for r in rows:
            w.write(r)


class LocalClassifier:
    def __init__(self, checkpoint: str):
        mod = __import__("intent_classifier.api", fromlist=["load_model", "predict_batch_with_model", "predict_batch"])
        if hasattr(mod, "load_model") and hasattr(mod, "predict_batch_with_model"):
            self.model = mod.load_model(checkpoint)
            self.predict_fn = mod.predict_batch_with_model
            self.mode = "model"
        elif hasattr(mod, "predict_batch"):
            self.model = checkpoint
            self.predict_fn = mod.predict_batch
            self.mode = "stateless"
        else:
            raise RuntimeError("intent_classifier.api missing required functions")

    def predict(self, items: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        if self.mode == "model":
            return self.predict_fn(self.model, items)
        return self.predict_fn(items)


class HTTPClassifier:
    def __init__(self, endpoint: str, checkpoint: str = ""):
        self.endpoint = endpoint
        self.checkpoint = checkpoint

    def predict(self, items: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        payload = {"items": items}
        if self.checkpoint:
            payload["checkpoint"] = self.checkpoint
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.endpoint, data=data, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=float(os.getenv("CLASSIFIER_TIMEOUT", "30"))) as resp:
            s = resp.read().decode("utf-8")
            j = json.loads(s)
            if isinstance(j, list):
                return j
            return j.get("items", [])


def load_backends(checkpoints: List[str], endpoint: str = ""):
    backends = []
    if endpoint:
        for ck in checkpoints:
            backends.append(HTTPClassifier(endpoint, ck))
    else:
        for ck in checkpoints:
            backends.append(LocalClassifier(ck))
    return backends


def batch_predict(backends, items: List[Dict[str, Any]], batch_size: int) -> List[List[Dict[str, Any]]]:
    all_preds = [[] for _ in backends]
    for i in range(0, len(items), batch_size):
        batch = [{"code": x["code"], "comment": x["comment"]} for x in items[i:i+batch_size]]
        for bi, be in enumerate(backends):
            preds = be.predict(batch)
            all_preds[bi].extend(preds)
    return all_preds


def vote_single(preds_per_model: List[Dict[str, Any]], vote: str, weights: List[float]) -> Tuple[str, float]:
    if vote == "weighted":
        scores = defaultdict(float)
        for p, w in zip(preds_per_model, weights):
            lab = str(p.get("intent", "")).lower()
            sc = float(p.get("score", 1.0))
            scores[lab] += w * sc
        if not scores:
            return "", 0.0
        lab = max(scores.items(), key=lambda x: x[1])[0]
        return lab, scores[lab]
    c = Counter()
    conf = defaultdict(list)
    for p in preds_per_model:
        lab = str(p.get("intent", "")).lower()
        sc = float(p.get("score", 1.0))
        c[lab] += 1
        conf[lab].append(sc)
    if not c:
        return "", 0.0
    top = c.most_common()
    best = [top[0][0]]
    best_cnt = top[0][1]
    for k, v in top[1:]:
        if v == best_cnt:
            best.append(k)
    if len(best) == 1:
        lab = best[0]
        return lab, sum(conf[lab]) / max(1, len(conf[lab]))
    avg_scores = {k: sum(conf[k]) / max(1, len(conf[k])) for k in best}
    lab = max(avg_scores.items(), key=lambda x: x[1])[0]
    return lab, avg_scores[lab]


def evaluate(golds: List[str], preds: List[str]) -> Dict[str, Any]:
    labels = sorted(list(set([g for g in golds if g != ""])) | set([p for p in preds if p != ""]))
    idx = {l: i for i, l in enumerate(labels)}
    tp = [0] * len(labels)
    fp = [0] * len(labels)
    fn = [0] * len(labels)
    correct = 0
    for g, p in zip(golds, preds):
        if g == p:
            correct += 1
            if g in idx:
                tp[idx[g]] += 1
        else:
            if p in idx:
                fp[idx[p]] += 1
            if g in idx:
                fn[idx[g]] += 1
    prec = []
    rec = []
    f1 = []
    for i in range(len(labels)):
        P = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0.0
        R = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0.0
        F = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
        prec.append(P); rec.append(R); f1.append(F)
    macro_p = sum(prec) / len(prec) if prec else 0.0
    macro_r = sum(rec) / len(rec) if rec else 0.0
    macro_f1 = sum(f1) / len(f1) if f1 else 0.0
    acc = correct / len(golds) if golds else 0.0
    per_label = {labels[i]: {"precision": prec[i], "recall": rec[i], "f1": f1[i]} for i in range(len(labels))}
    return {"accuracy": acc, "macro_precision": macro_p, "macro_recall": macro_r, "macro_f1": macro_f1, "per_label": per_label}


def main():
    ap = argparse.ArgumentParser()import os
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

    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--checkpoints", required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--endpoint", default="")
    ap.add_argument("--vote", choices=["majority", "weighted"], default="majority")
    ap.add_argument("--weights", default="")
    ap.add_argument("--code_field", default="code")
    ap.add_argument("--comment_field", default="comment")
    ap.add_argument("--label_field", default="label")
    args = ap.parse_args()

    checkpoints = [x.strip() for x in args.checkpoints.split(",") if x.strip()]
    if not checkpoints:
        raise SystemExit("no checkpoints")
    if args.vote == "weighted":
        if args.weights:
            weights = [float(x) for x in args.weights.split(",")]
            if len(weights) != len(checkpoints):
                raise SystemExit("weights size mismatch")
        else:
            weights = [1.0] * len(checkpoints)
    else:
        weights = [1.0] * len(checkpoints)

    backends = load_backends(checkpoints, args.endpoint)
    items = read_items(args.input, args.code_field, args.comment_field)
    preds_models = batch_predict(backends, items, args.batch_size)

    merged = []
    golds = []
    for i in range(len(items)):
        per_model = [preds_models[m][i] for m in range(len(preds_models))]
        lab, conf = vote_single(per_model, args.vote, weights)
        raw = items[i]["_raw"]
        out = dict(raw)
        out["pred_intent"] = lab
        out["pred_conf"] = conf
        out["per_model"] = per_model
        merged.append(out)
        golds.append(str(raw.get(args.label_field, "")).lower())

    write_items(args.output, merged)

    if any(golds):
        pred_labels = [m["pred_intent"] for m in merged]
        metrics = evaluate(golds, pred_labels)
        metrics_path = os.path.splitext(args.output)[0] + ".metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
