import os
import json
import jsonlines
import random
import numpy as np
import torch

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)import os
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
