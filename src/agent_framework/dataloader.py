import argparse
import os
import random
from typing import List, Dict, Any, Optional
from utils import read_jsonl, write_jsonl, normalize_text, set_seed

INTENTS = {"what", "why", "done", "property"}


def _norm_label(x: str) -> str:
    if not isinstance(x, str):
        return ""
    y = normalize_text(x)
    y = y.replace("-", "_")
    if y in {"how_it_is_done", "how-it-is-done"}:
        return "done"
    if y in INTENTS:
        return y
    return y


def _keep_record(code: str, comment: str, label: str, allow: Optional[List[str]] = None) -> bool:
    if not code or not comment or not label:
        return False
    if allow:
        s = set(_norm_label(a) for a in allow)
        if _norm_label(label) not in s:
            return False
    return True


def _build_record(obj: Dict[str, Any], code_f: str, cmt_f: str, lbl_f: str, keep_fields: Optional[List[str]]) -> Optional[Dict[str, Any]]:
    code = obj.get(code_f, "")
    comment = obj.get(cmt_f, "")
    label = _norm_label(obj.get(lbl_f, ""))
    if not _keep_record(code, comment, label):
        return None
    rec = {"code": code, "comment": comment, "label": label}
    if keep_fields:
        for k in keep_fields:
            if k in obj:
                rec[k] = obj[k]
    return rec


def load_and_convert(path: str, code_field: str, comment_field: str, label_field: str, keep_fields: Optional[List[str]] = None, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    rows = read_jsonl(path)
    out: List[Dict[str, Any]] = []
    for r in rows:
        rec = _build_record(r, code_field, comment_field, label_field, keep_fields)
        if rec is None:
            continue
        out.append(rec)
        if max_samples and len(out) >= max_samples:
            break
    return out


def filter_by_labels(rows: List[Dict[str, Any]], allow: Optional[List[str]]) -> List[Dict[str, Any]]:
    if not allow:
        return rows
    s = set(_norm_label(a) for a in allow)
    return [r for r in rows if r.get("label") in s]


def deduplicate(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for r in rows:
        key = (r.get("code", ""), r.get("comment", ""), r.get("label", ""))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def split_rows(rows: List[Dict[str, Any]], train_ratio: float, dev_ratio: float, seed: int) -> Dict[str, List[Dict[str, Any]]]:
    set_seed(seed)
    n = len(rows)
    idx = list(range(n))
    random.shuffle(idx)
    n_train = int(n * train_ratio)
    n_dev = int(n * dev_ratio)
    train_idx = idx[:n_train]
    dev_idx = idx[n_train:n_train + n_dev]
    test_idx = idx[n_train + n_dev:]
    return {
        "train": [rows[i] for i in train_idx],
        "dev": [rows[i] for i in dev_idx],
        "test": [rows[i] for i in test_idx],
    }


def run(args):
    rows = load_and_convert(
        path=args.input,
        code_field=args.code_field,
        comment_field=args.comment_field,
        label_field=args.label_field,
        keep_fields=args.keep_fields,
        max_samples=args.max_samples,
    )
    rows = filter_by_labels(rows, args.filter_labels)
    rows = deduplicate(rows)
    if args.shuffle:
        set_seed(args.seed)
        random.shuffle(rows)
    os.makedirs(args.output_dir, exist_ok=True)
    if args.split:
        parts = split_rows(rows, args.train_ratio, args.dev_ratio, args.seed)
        write_jsonl(os.path.join(args.output_dir, "train.jsonl"), parts["train"])
        write_jsonl(os.path.join(args.output_dir, "dev.jsonl"), parts["dev"])
        write_jsonl(os.path.join(args.output_dir, "test.jsonl"), parts["test"])
    else:
        write_jsonl(os.path.join(args.output_dir, "dataset.jsonl"), rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--code_field", default="code")
    ap.add_argument("--comment_field", default="comment")
    ap.add_argument("--label_field", default="label")
    ap.add_argument("--keep_fields", nargs="*", default=["parsed_results", "ori_code"])
    ap.add_argument("--filter_labels", nargs="*", default=["what", "why", "done", "property"])
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--split", action="store_true")
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--dev_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shuffle", action="store_true")
    args = ap.parse_args()
    run(args)