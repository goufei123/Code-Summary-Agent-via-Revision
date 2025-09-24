import os
import json
import jsonlines
import argparse
import logging
from tqdm import tqdm
from evaluation.evall.bleu import corpus_bleu
from evaluation.evall.rouge import Rouge
from evaluation.evall.meteor import Meteor
from openai import OpenAI
from collections import defaultdict
from tool_module import get_examples, get_context
from langgraph.graph import StateGraph, END


INTENT_NAME = {"what": "What", "done": "How-it-is-done", "property": "Property", "why": "Why"}

CLS_PROMPT = {
    'what': 'Please generate a short comment in one sentence describing what this function does and its primary purpose:',
    'property': 'Please generate a short comment in one sentence highlighting a key property of this function:',
    'done': 'Please generate a short comment in one sentence explaining how this function works and what it does internally:',
    'why': 'Please generate a short comment in one sentence explaining why this function work:'
}

MODEL_CONFIG = {
    "gpt": {"type": "api", "model": "gpt-4o", "base_url": "https://api.openai.com/v1"},
    "deepseek": {"type": "api", "model": "deepseek-chat", "base_url": "https://api.deepseek.com"}
}

def get_model_client(args):
    config = MODEL_CONFIG[args.model]
    if config["type"] == "api":
        if args.model == "deepseek":
            return OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url=config["base_url"]), config["model"]
        else:
            return OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=config["base_url"]), config["model"]
    raise ValueError("Unsupported model type")

def eval_accuracies(hypotheses, references):
    _, bleu, _ = corpus_bleu(hypotheses, references)
    rouge = Rouge().compute_score(references, hypotheses)[0]
    meteor_calc = Meteor()
    meteor = meteor_calc.compute_score(references, hypotheses)[0]
    meteor_calc.close()
    return bleu * 100, rouge * 100, meteor * 100

def save_predictions(path, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with jsonlines.open(path, 'w') as writer:
        for r in records:
            writer.write(r)

def chat_complete(client, model_name, messages, temperature):
    r = client.chat.completions.create(model=model_name, messages=messages, max_tokens=256, temperature=temperature)
    return r.choices[0].message.content.strip()

def initial_summary(client, model_name, intent, code, temperature):
    name = INTENT_NAME.get(intent, intent)
    sys = "You write precise one-sentence code comments aligned with a requested intent."
    usr = f"Intent: {name}\nCode:\n{code}\nReturn only one sentence."
    return chat_complete(client, model_name, [{"role": "system", "content": sys}, {"role": "user", "content": usr}], temperature)

def assess_summary(client, model_name, intent, code, summary, temperature):
    sys = "You are an evaluator. Output a JSON with numeric fields intent_alignment, content_adequacy, usefulness scored from 1 to 5."
    usr = json.dumps({"intent": INTENT_NAME.get(intent, intent), "code": code, "summary": summary})
    out = chat_complete(client, model_name, [{"role": "system", "content": sys}, {"role": "user", "content": usr}], temperature)
    try:
        j = None
        s = out
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            j = json.loads(s[start:end+1])
        else:
            j = json.loads(s)
        a = float(j.get("intent_alignment", 0))
        c = float(j.get("content_adequacy", 0))
        u = float(j.get("usefulness", 0))
        a = max(1.0, min(5.0, a))
        c = max(1.0, min(5.0, c))
        u = max(1.0, min(5.0, u))
        avg = (a + c + u) / 3.0
        return avg, {"intent_alignment": a, "content_adequacy": c, "usefulness": u}
    except Exception:
        return 0.0, {"intent_alignment": 0.0, "content_adequacy": 0.0, "usefulness": 0.0}

def plan_revisions(client, model_name, intent, code, summary, scores, temperature):
    sys = "You are a planner. Given intent, code, current summary and scores, propose up to 3 concise revision plans. Output JSON {\"plans\": [..]}"
    payload = {"intent": INTENT_NAME.get(intent, intent), "scores": scores, "code": code, "summary": summary}
    out = chat_complete(client, model_name, [{"role": "system", "content": sys}, {"role": "user", "content": json.dumps(payload)}], temperature)
    try:
        s = out
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            j = json.loads(s[start:end+1])
        else:
            j = json.loads(s)
        plans = j.get("plans", [])
        plans = [str(p).strip() for p in plans if str(p).strip()]
        return plans[:3]
    except Exception:
        return []

def build_supply_info(intent, code, parsed_results, need_context=False, need_examples=False):
    parts = []
    if need_examples:
        ex = get_examples(intent, code) or []
        ex_texts = []
        for i, e in enumerate(ex[:3]):
            ec = e.get("code", "")
            cm = e.get("comment", "")
            ex_texts.append(f"Example {i+1}:\nCode:\n{ec}\nComment:\n{cm}")
        ex_block = "\n\n".join(ex_texts)
        if ex_block:
            parts.append(ex_block)
    if need_context:
        ctx = get_context(parsed_results if parsed_results else {"code": code}) or ""
        if ctx:
            parts.append(ctx)
    return "\n\n".join(parts)

def revise_summary(client, model_name, intent, code, prev_summary, plans, supply_info, temperature):
    name = INTENT_NAME.get(intent, intent)
    sys = "You revise code comments. Rewrite into one sentence aligned with the intent, following the plans and using supply info when helpful. Return only the revised sentence."
    data = {"intent": name, "code": code, "previous_summary": prev_summary, "plans": plans, "supply_info": supply_info}
    return chat_complete(client, model_name, [{"role": "system", "content": sys}, {"role": "user", "content": json.dumps(data)}], temperature)

class AgentState(dict):
    pass

def build_agent_graph(client, model_name, temperature, threshold, max_rounds):
    def node_generate(state: AgentState):
        s = state.copy()
        s["summary"] = initial_summary(client, model_name, s["intent"], s["code"], temperature)
        s["round"] = 0
        return s

    def node_evaluate(state: AgentState):
        s = state.copy()
        avg, scores = assess_summary(client, model_name, s["intent"], s["code"], s.get("summary", ""), temperature)
        s["avg"] = avg
        s["scores"] = scores
        return s

    def node_plan(state: AgentState):
        s = state.copy()
        plans = plan_revisions(client, model_name, s["intent"], s["code"], s.get("summary", ""), s.get("scores", {}), temperature)
        s["plans"] = plans
        need_ctx = False
        need_ex = False
        for p in plans or []:
            t = str(p).lower()
            if any(k in t for k in ["context", "content", "doc", "callee", "caller"]):
                need_ctx = True
            if any(k in t for k in ["example", "few-shot", "few shot", "retriev"]):
                need_ex = True
        s["need_context"] = need_ctx
        s["need_examples"] = need_ex
        return s

    def node_supply(state: AgentState):
        s = state.copy()
        supply = build_supply_info(s["intent"], s["code"], s.get("parsed_results"), s.get("need_context", False), s.get("need_examples", False))
        s["supply_info"] = supply
        return s

    def node_revise(state: AgentState):
        s = state.copy()
        new_summary = revise_summary(client, model_name, s["intent"], s["code"], s.get("summary", ""), s.get("plans", []), s.get("supply_info", ""), temperature)
        s["summary"] = new_summary
        s["round"] = int(s.get("round", 0)) + 1
        return s

    def should_stop(state: AgentState):
        avg = float(state.get("avg", 0.0))
        rnd = int(state.get("round", 0))
        if avg >= threshold or rnd >= max_rounds:
            return True
        return False

    graph = StateGraph(AgentState)
    graph.add_node("generate", node_generate)
    graph.add_node("evaluate", node_evaluate)
    graph.add_node("plan", node_plan)
    graph.add_node("supply", node_supply)
    graph.add_node("revise", node_revise)
    graph.set_entry_point("generate")
    graph.add_edge("generate", "evaluate")
    graph.add_conditional_edges(
        "evaluate",
        lambda s: "end" if should_stop(s) else "continue",
        {"end": END, "continue": "plan"},
    )
    graph.add_edge("plan", "supply")
    graph.add_edge("supply", "revise")
    graph.add_edge("revise", "evaluate")
    return graph.compile()


def run_agent_graph(client, model_name, intent, code, parsed_results, temperature, max_rounds, threshold):
    app = build_agent_graph(client, model_name, temperature, threshold, max_rounds)
    state = AgentState({"intent": intent, "code": code, "parsed_results": parsed_results})
    out = app.invoke(state)
    return out.get("summary", ""), float(out.get("avg", 0.0)), out.get("scores", {})

def generate(args, data):
    client, model_name = get_model_client(args)
    intents = ["what", "done", "property", "why"]
    intent_scores = {k: {'bleu': 0, 'meteor': 0, 'rouge_l': 0, 'count': 0} for k in intents}
    save_path = os.path.join(args.output_dir, f"agent_{args.model}.jsonl")
    all_records = []
    for idx, obj in tqdm(enumerate(data)):
        if idx >= args.test_number:
            break
        label = obj["comment"]
        code = obj["code"]
        cls = obj["label"]
        if cls not in intents:
            continue
        pred, avg, scores = run_agent_graph(client, model_name, cls, code, obj.get("parsed_results"), args.temperature, args.max_rounds, args.threshold)
        bleu, rouge, meteor = eval_accuracies({0: [pred.strip().split('\n')[0]]}, {0: [label.strip().split('\n')[0]]})
        intent_scores[cls]['bleu'] += bleu
        intent_scores[cls]['meteor'] += meteor
        intent_scores[cls]['rouge_l'] += rouge
        intent_scores[cls]['count'] += 1
        record = {"intent": cls, "code": code, "ori_code": obj.get("ori_code", code), "label": label, "prediction": pred, "assessor_avg": avg, "assessor_scores": scores, "bleu": bleu, "rouge": rouge, "meteor": meteor}
        all_records.append(record)
    save_predictions(save_path, all_records)
    for cls in intents:
        c = intent_scores[cls]['count']
        if c == 0:
            print(f"{cls.upper()} - No samples")
            continue
        b = intent_scores[cls]['bleu'] / c
        m = intent_scores[cls]['meteor'] / c
        r = intent_scores[cls]['rouge_l'] / c
        print(f"{cls.lower()} - BLEU: {b:.2f}, ROUGE-L: {r:.2f}, METEOR: {m:.2f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["gpt", "deepseek"], default="deepseek")
    parser.add_argument("--prompt_filename", default="./dataset/cls_examples_test_all.jsonl", type=str)
    parser.add_argument("--output_dir", default="./output/eval_result/", type=str)
    parser.add_argument("--temperature", default=0.75, type=float)
    parser.add_argument("--test_number", default=15000, type=int)
    parser.add_argument("--max_rounds", default=3, type=int)
    parser.add_argument("--threshold", default=4.0, type=float)
    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
    data = []
    with jsonlines.open(args.prompt_filename) as reader:
        for obj in reader:
            data.append(obj)
    generate(args, data)

if __name__ == '__main__':
    main()