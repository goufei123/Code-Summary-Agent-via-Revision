import os
import jsonlines
import argparse
import logging
from tqdm import tqdm
from evaluation.evall.bleu import corpus_bleu
from evaluation.evall.rouge import Rouge
from evaluation.evall.meteor import Meteor
from openai import OpenAI
# from model import CodeLLAMA
import re
# from bert_score import score
from collections import defaultdict

CLS_PROMPT = {
    'what': 'Please generate a short comment in one sentence describing what this function does and its primary purpose:',
    'property': 'Please generate a short comment in one sentence highlighting a key property of this function:',
    'done': 'Please generate a short comment in one sentence explaining how this function works and what it does internally:',
    'why': 'Please generate a short comment in one sentence explaining why this function work:'
}

FEW_SHOT_SAMPLE = {
    'what': [
        "Gets the codec info from terms.",
        "Wraps an ObservableSource into an Observable if not already an Observable.",
        "Creates an UnicastProcessor with the given internal buffer capacity hint.",
        "Removes all handlers and resets to default behavior.",
        "Child Observers will observe the events of the ConnectableObservable on the specified scheduler."
    ],
    'done': [
        "Creates an UnicastProcessor with the given internal buffer capacity hint and a callback for the case when the single Subscriber cancels its subscription.",
        "Maps a scalar value into a Publisher and emits its values.",
        "Sorts the given list using the given comparator.",
        "Reset all custom handlers to their default values.",
        "Extracts the first emitted value and maps it into another Publisher."
    ],
    'property': [
        "Returns a List of entries in the map sorted by key.",
        "Ensures the returned collection is immutable.",
        "Returns a stream sorted in descending order.",
        "The resulting map maintains insertion order.",
        "Sorting preserves duplicate values with original order."
    ],
    'why': [
        "Americanize and print the command line arguments. This main method is just for debugging.",
        "Used to standardize input before main logic for easier debugging.",
        "Acts as a stub for launching the application in test environments.",
        "Helps log debug output to verify initialization sequence.",
        "Used to quickly inspect function behavior during development."
    ]
}

MODEL_CONFIG = {
    "gpt": {
        "type": "api",
        "model": "gpt-4o",
        "base_url": "https://api.openai.com/v1",
        "api_key": "sk-proj-H5sIv2NWZfGdGTppPtmBT3BlbkFJsigPiyhyP1ns5zpOF8y0"
    },
    "deepseek": {
        "type": "api",
        "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com",
        "api_key": "sk-78a92b74eaec460391d3dd5b5bc6fef3"
    },
    "codellama": {
        "type": "local",
        "model_path": "/root/GYF/Models/codellama"
    }
}


def get_model_client(args):
    config = MODEL_CONFIG[args.model]
    if config["type"] == "api":
        return OpenAI(api_key=config["api_key"], base_url=config["base_url"]), config["model"]
    elif config["type"] == "local":
        args.model_name_or_path = config["model_path"]
        return CodeLLAMA(args), None
    else:
        raise ValueError(f"Unsupported model type: {config['type']}")

def eval_accuracies(hypotheses, references):
    _, bleu, _ = corpus_bleu(hypotheses, references)
    rouge = Rouge().compute_score(references, hypotheses)[0]
    meteor_calc = Meteor()
    meteor = meteor_calc.compute_score(references, hypotheses)[0]
    meteor_calc.close()
    return bleu * 100, rouge * 100, meteor * 100


# Helper: Save predictions to JSONL
def save_predictions(path, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with jsonlines.open(path, 'w') as writer:
        for r in records:
            writer.write(r)

# Helper: Batch BERTScore from JSONL file
def batch_bertscore_from_json(path, model_type='roberta-large'):
    preds, labels, intents = [], [], []
    with jsonlines.open(path, 'r') as reader:
        for obj in reader:
            # keep one-line summary as used elsewhere
            pred = str(obj.get('prediction', '')).strip().split('\n')[0]
            label = str(obj.get('label', '')).strip().split('\n')[0]
            preds.append(pred)
            labels.append(label)
            intents.append(obj.get('intent', 'unknown'))
    if len(preds) == 0:
        return [], 0.0, {}
    P, R, F1 = score(preds, labels, lang='en', verbose=False, model_type=model_type)
    f1_list = [round(x.item(), 4) * 100 for x in F1]
    overall = sum(f1_list) / len(f1_list)
    by_intent = defaultdict(list)
    for i, intent in enumerate(intents):
        by_intent[intent].append(f1_list[i])
    by_intent_avg = {k: (sum(v) / len(v) if len(v) else 0.0) for k, v in by_intent.items()}
    return f1_list, overall, by_intent_avg

def generate_score(pred, label):
    pred = pred.strip().split('\n')[0]
    bleu, rouge, meteor = eval_accuracies({0: [pred]}, {0: [label]})
    return bleu, rouge, meteor

def build_prompt(args, code, parsed_results=None, examples=None):
    task_prompt = CLS_PROMPT[args.cls_label]

    if args.prompt_type.startswith("few-shot"):
        # Support "few-shot", "few-shot3", "few-shot5", etc.
        try:
            num_examples = int(args.prompt_type.replace("few-shot", ""))
        except ValueError:
            num_examples = 1
        # Limit to max 5 examples or available examples
        num_examples = max(1, min(num_examples, len(FEW_SHOT_SAMPLE[args.cls_label]), 5))
        shots = [f"Example {i+1}: {s}" for i, s in enumerate(FEW_SHOT_SAMPLE[args.cls_label][:num_examples])]
        few_shot_block = "\n".join(shots)
        return f"{task_prompt}\n{few_shot_block}\nCode: {code}"

    elif args.prompt_type == "cot":
        prompt1 = f'''{task_prompt}
Code:
{code}

Question：
1. What is the name of the function?
2. What are the input parameters that are being accepted by the function?
3. What is the expected output or return value of the function?
4. Are there any specific requirements or constraints for using this function?
5. Does the function have any additional dependencies or external requirements?

Please answer the above questions but only return a short comment in one sentence describing the function. ONLY Comment'''
        return prompt1

    elif args.prompt_type == "expert":
        expert_prompt = "You are an expert developer. Summarize the function by explaining its purpose, parameters, and return value."
        expert_example = [
            {'Instruction': 'Make a list of 5 possible effects of deforestation.',
             'Description': 'You are an environmental scientist with a specialization in the study of ecosystems and their interactions with human activities. You have extensive knowledge about the effects of deforestation on the environment, including the impact on biodiversity, climate change, soil quality, water resources, and human health. Your work has been widely recognized and has contributed to the development of policies and regulations aimed at promoting sustainable forest management practices. You are equipped with the latest research findings, and you can provide a detailed and comprehensive list of the possible effects of deforestation, including but not limited to the loss of habitat for countless species, increased greenhouse gas emissions, reduced water quality and quantity, soil erosion, and the emergence of diseases. Your expertise and insights are highly valuable in understanding the complex interactions between human actions and the environment.'
             },
            {'Instruction': 'Identify a descriptive phrase for an eclipse.',
             'Description': 'You are an astronomer with a deep understanding of celestial events and phenomena. Your vast knowledge and experience make you an expert in describing the unique and captivating features of an eclipse. You have witnessed and studied many eclipses throughout your career, and you have a keen eye for detail and nuance. Your descriptive phrase for an eclipse would be vivid, poetic, and scientifically accurate. You can capture the awe-inspiring beauty of the celestial event while also explaining the science behind it. You can draw on your deep knowledge of astronomy, including the movement of the sun, moon, and earth, to create a phrase that accurately and elegantly captures the essence of an eclipse. Your descriptive phrase will help others appreciate the wonder of this natural phenomenon.'
             },
            {'Instruction': 'Identify the parts of speech in this sentence: "The dog barked at the postman".',
             'Description': 'You are a linguist, well-versed in the study of language and its structures. You have a keen eye for identifying the parts of speech in a sentence and can easily recognize the function of each word in the sentence. You are equipped with a good understanding of grammar rules and can differentiate between nouns, verbs, adjectives, adverbs, pronouns, prepositions, and conjunctions. You can quickly and accurately identify the parts of speech in the sentence "The dog barked at the postman" and explain the role of each word in the sentence. Your expertise in language and grammar is highly valuable in analyzing and understanding the nuances of communication.'
             }
        ]
        history_prompt = [(expert_prompt + ex['Instruction'], ex['Description']) for ex in expert_example]
        return f"{task_prompt}\nCode:\n{code}", history_prompt

    elif args.prompt_type == "zero-shot":
        return f"{task_prompt}\nCode: {code}"

    elif args.prompt_type == "react":
        return None  # Prompt is not used for react; handled externally

    elif args.prompt_type == "content-aware":
        texts = []
        if parsed_results:
            if parsed_results.get("comments"):
                try:
                    raw_comment = parsed_results["comments"]
                    if isinstance(raw_comment, str):
                        raw_comment = raw_comment.strip()
                        lines = raw_comment.splitlines()
                        allowed_tags = {"@param", "@return", "@throws", "@since", "@deprecated"}
                        extracted = []
                        current = None
                        for line in lines:
                            stripped = line.strip()
                            if any(stripped.startswith(tag) for tag in allowed_tags):
                                current = stripped
                                extracted.append(current)
                            elif current and (line.startswith(" ") or line.startswith("\t")):
                                # 是缩进行，属于当前 tag 的扩展
                                extracted[-1] += " " + stripped
                            else:
                                current = None
                        # 清除 {@code xxx} / {@link xxx}
                        extracted = [re.sub(r'\{@\w+\s+([^}]+)\}', r'\1', x) for x in extracted]
                        if extracted:
                            texts.append("Docstring:\n" + "\n".join(extracted))
                except Exception:
                    pass

            if parsed_results.get("target"):
                repo_class = parsed_results["target"]
                dot_index = repo_class.rfind('.')
                if dot_index != -1:
                    class_name = repo_class[:dot_index].split('.')[-1]
                    texts.append(f"The function comes from: {class_name} class.")

            callee = parsed_results.get("callees", [])
            callee_texts = []
            for i in range(0, len(callee), 3):
                group = callee[i:i+3]
                if len(group) >= 3 and group[1] and group[2]:
                    code_seg = group[2]
                    pattern = r'public\s+\w+\s+\w+\([^)]*\)[^}]*\{[^}]*\}'
                    try:
                        if isinstance(code_seg, str):
                            matches = re.findall(pattern, code_seg, flags=re.DOTALL)
                            if matches:
                                code_seg = re.sub(r'\s+', ' ', matches[0].strip())
                            else:
                                code_seg = re.sub(r'^\s+', '', code_seg, flags=re.MULTILINE).strip()
                        else:
                            code_seg = "[Unparsable code snippet]"
                    except Exception:
                        code_seg = "[Unparsable code snippet]"
                    callee_texts.append(f"function `{group[1]}` which is `{code_seg}`")
            if callee_texts:
                texts.append("The function calls: " + ", ".join(callee_texts))

            caller = parsed_results.get("callers", [])
            caller_texts = []
            for i in range(0, len(caller), 3):
                group = caller[i:i+3]
                if len(group) >= 3 and group[0]:
                    if isinstance(group[0], str):
                        try:
                            comment = re.sub(r'^\s*', '', group[0], flags=re.MULTILINE)
                            comment = re.sub(r'@.*', '', comment, flags=re.DOTALL).strip()
                        except Exception:
                            comment = "[Unparsable caller comment]"
                    else:
                        comment = "[Unparsable caller comment]"
                    caller_texts.append(f"function `{group[2]}` which is used to {comment}")
            if caller_texts:
                texts.append("The function is used by: " + ", ".join(caller_texts))

            texts.append("Based on the above information,")

        context = "\n\n".join(texts)
        return f"{context}\n\n{task_prompt}\nCode:\n{code}"

    elif args.prompt_type == "example-aware":
        if not examples or len(examples) == 0:
            return f"{task_prompt}\nCode:\n{code}"

        shots = []
        for i, ex in enumerate(examples[:3]):  # 最多使用3个few-shot样例
            if 'code' in ex and 'comment' in ex:
                shots.append(f"Example {i+1}:\nCode:\n{ex['code']}\nComment:\n{ex['comment']}")
        few_shot_block = "\n\n".join(shots)
        return f"Here are some examples:\n\n{few_shot_block}\n\n{task_prompt}\nCode: {code}"

    else:
        raise ValueError(f"Unsupported prompt_type: {args.prompt_type}")

def callee_generate_task(args, query, results):
    model, model_name = get_model_client(args)
    intents = ["what", "done", "property", "why"]
    intent_scores = {intent: {'bleu': 0, 'meteor': 0, 'rouge_l': 0, 'count': 0} for intent in intents}

    save_path = f"./output/human_eval/{args.prompt_type}.json"
    collected_samples = {intent: [] for intent in intents}
    max_per_class = 2000
    all_records = []

    for idx, (label, code) in tqdm(enumerate(query)):
        if idx >= args.test_number:
            break

        cls = results["cls_label"][idx]
        if cls not in intents:
            continue
        if len(collected_samples[cls]) >= max_per_class:
            continue

        args.cls_label = cls
        parsed_results = results["parsed_results"][idx]
        examples = results["examples"][idx]
        # Adjust for expert mode to unpack tuple
        if args.prompt_type == "expert":
            prompt, history_prompt = build_prompt(args, code, parsed_results=parsed_results, examples=examples)
        elif args.prompt_type == "react":
            prompt = None
            history_prompt = []
        else:
            prompt = build_prompt(args, code, parsed_results=parsed_results, examples=examples)
            history_prompt = []

        # ReAct branch
        if args.prompt_type == "react":
            from langchain.agents import AgentExecutor
            from langchain.agents.react.agent import create_react_agent
            from langchain_community.chat_models import ChatOpenAI
            from langchain.tools import tool
            from langchain.prompts import PromptTemplate

            task_prompt = CLS_PROMPT[args.cls_label]

            @tool
            def extract_docstring(doc: str) -> str:
                """Extract @param, @return and similar tags from the docstring of the given code."""
                import re
                allowed_tags = {"@param", "@return", "@throws"}
                lines = doc.splitlines()
                extracted = []
                current = None
                for line in lines:
                    stripped = line.strip()
                    if any(stripped.startswith(tag) for tag in allowed_tags):
                        current = stripped
                        extracted.append(current)
                    elif current and (line.startswith(" ") or line.startswith("\t")):
                        extracted[-1] += " " + stripped
                    else:
                        current = None
                return "\n".join(extracted) if extracted else "No structured docstring found."

            llm = ChatOpenAI(
                model="gpt-4",
                temperature=args.temperature,
                api_key="sk-proj-H5sIv2NWZfGdGTppPtmBT3BlbkFJsigPiyhyP1ns5zpOF8y0"
            )
            tools = [extract_docstring]
            prompt_template = PromptTemplate.from_template(
                """You are a professional code summarizer.
You can use the following tools:
{tools}

To solve the task, follow this format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original question

Begin!

Question: {input}
{agent_scratchpad}"""
            )
            agent = create_react_agent(llm=llm, tools=tools, prompt=prompt_template)
            executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
            comment = parsed_results.get("comments", "")
            message = executor.invoke({
                "input": f"Question: {task_prompt}\nDoc:\n{comment}\nCode:\n{code}"
            })["output"]
        else:
            if MODEL_CONFIG[args.model]["type"] == "api":
                messages = [{"role": "system", "content": "You are a helpful assistant."}]
                if args.prompt_type == "expert":
                    for hist_input, hist_reply in history_prompt:
                        messages.append({"role": "user", "content": hist_input})
                        messages.append({"role": "assistant", "content": hist_reply})
                messages.append({"role": "user", "content": prompt})

                response = model.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=150,
                    temperature=args.temperature
                )
                message = response.choices[0].message.content.strip()
            else:
                message = model.ask(
                    input=prompt,
                    history=history_prompt,
                    system_prompt="You are a helpful assistant."
                ) if args.prompt_type == "expert" else model.ask(
                    prompt, history=[],
                    system_prompt="You are an AI assistant that summarizes code functions."
                )

        ori_code = results["ori_code"][idx] if "ori_code" in results else code
        bleu, rouge, meteor = generate_score(message, label)
        intent_scores[cls]['bleu'] += bleu
        intent_scores[cls]['meteor'] += meteor
        intent_scores[cls]['rouge_l'] += rouge
        intent_scores[cls]['count'] += 1
        record = {
            "intent": cls,
            "code": code,
            "ori_code": ori_code,
            "label": label,
            "prediction": message,
            "bleu": bleu,
            "rouge": rouge,
            "meteor": meteor
        }
        collected_samples[cls].append(record)
        all_records.append(record)

    # Save all predictions first
    save_predictions(save_path, all_records)

    # Batch compute BERTScore from saved file
    try:
        _, bert_overall, bert_by_intent = batch_bertscore_from_json(save_path, model_type=getattr(args, 'bert_model', 'roberta-large'))
    except Exception as e:
        bert_overall, bert_by_intent = 0.0, {}
        print(f"[Warning] BERTScore batch computation failed: {e}")

    for cls in intents:
        count = intent_scores[cls]['count']
        if count == 0:
            print(f"{cls.upper()} - No samples found.")
            continue
        bleu = intent_scores[cls]['bleu'] / count
        meteor = intent_scores[cls]['meteor'] / count
        rouge = intent_scores[cls]['rouge_l'] / count
        bert = bert_by_intent.get(cls, 0.0)
        print(f"{cls.lower()} - BLEU: {bleu:.2f}, ROUGE-L: {rouge:.2f}, METEOR: {meteor:.2f}, BERTScore(F1): {bert:.2f}")
    print(f"overall - BERTScore(F1): {bert_overall:.2f}")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["gpt", "deepseek", "codellama"], default='deepseek')
    parser.add_argument("--task", default='summary_java')
    parser.add_argument("--prompt_filename", default='./output/cls_examples_test_all.jsonl', type=str)
    parser.add_argument("--output_dir", default='./output/eval_result/', type=str)
    parser.add_argument("--gpu_input", type=int, default=0, help="Index of the GPU to use for inference")
    parser.add_argument("--temperature", default=0.7, type=float)
    parser.add_argument("--test_number", default=15000, type=int)
    parser.add_argument("--max_new_tokens", default=18000, type=int)
    parser.add_argument("--top_k", default=50, type=int, help="Top-k sampling")
    parser.add_argument("--top_p", default=0.95, type=float, help="Top-p (nucleus) sampling")
    parser.add_argument("--prompt_type", type=str,  default="few-shot3", help='few-shot[N], cot, expert, zero-shot, content-aware, example-aware, react')
    parser.add_argument("--bert_model", default='roberta-large', type=str, help='HF model name or local path for BERTScore')
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
    args.logger = logging.getLogger(__name__)

    examples, query, parsed_results, cls_label = [], [], [], []
    ori_code_list = []

    with jsonlines.open(args.prompt_filename) as reader:
        for obj in reader:
            query.append((obj['comment'], obj['code']))
            parsed_results.append(obj['parsed_results'])
            cls_label.append(obj['label'])
            examples.append(obj.get('examples', []))  # 安全读取
            ori_code_list.append(obj.get('ori_code', obj.get('code', '')))

    results = {
        "parsed_results": parsed_results,
        "cls_label": cls_label,
        "examples": examples,
        "ori_code": ori_code_list,
    }

    callee_generate_task(args, query, results)

if __name__ == '__main__':
    main()