import os
import argparse
import random
import torch

from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, set_seed
import numpy as np
import pandas as pd
import json
from functools import partial
import asyncio
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

seed = 42

# Python & NumPy
random.seed(seed)
np.random.seed(seed)

# Torch
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Transformers
set_seed(seed)

# 保证CUDA确定性
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="score the conversations")

    parser.add_argument(
        "--proportion",
        type = float,
        default = 0.5
    )
    parser.add_argument(
        "--api_key",
        type = str,
        required=True
    )
    parser.add_argument(
        "--end_point",
        type = str,
        required=True
    )
    parser.add_argument(
        "--llm_model_name",
        type = str,
        required=True
    )
    parser.add_argument(
        "--bert_path",
        type = str,
        required=True
    )
    parser.add_argument(
        "--source_file",
        type = str,
        required=True
    )
    parser.add_argument(
        "--last_token_num",
        type = int,
        default = 512
    )

    args = parser.parse_args()
    return args

args = parse_args()

def flatten_dialogue(messages):
    flatten_msgs = ''
    
    for i, msg in enumerate(messages):
        if msg['role'] =='assistant':
            label = '[ASSISTANT]'
        else:
            label = '[USER]'

        current_msg = msg['content']
    
        flatten_msgs = flatten_msgs + label + current_msg
    
    return flatten_msgs

def prepare_data(json_path):
    test_data = []
    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                new_line = {}
                json_line = json.loads(line)
                test_data.append(flatten_dialogue(json_line['messages']))
    return test_data

def preprocess_tokens(example, N=512, direction="tail", max_length=512):
    tokens = tokenizer.tokenize(example["text"])
    
    # 最大 token 长度 = max_length - 2 (CLS 和 SEP)
    N = min(N, max_length - 2)
    
    if direction == "tail":
        tokens = tokens[-N:]
    else:
        tokens = tokens[:N]
    
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]
    
    attention_mask = [1] * len(input_ids)
    
    # padding 到固定长度
    pad_len = max_length - len(input_ids)
    if pad_len > 0:
        input_ids += [tokenizer.pad_token_id] * pad_len
        attention_mask += [0] * pad_len
    
    return {"input_ids": input_ids, "attention_mask": attention_mask}



model_path = args.bert_path  
model = AutoModelForSequenceClassification.from_pretrained(model_path)

tokenizer = AutoTokenizer.from_pretrained(model_path)

# 加载 test.jsonl
#test_data = load_dataset("json", data_files="test.jsonl")["train"]
test_data = prepare_data(args.source_file)
data_dict = {"text": test_data}
testdata = Dataset.from_dict(data_dict)

# 预处理（与训练一致）
def preprocess(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)
test_encoded = testdata.map(preprocess, batched=True)
print(test_encoded)
process_fn  = partial(preprocess_tokens, N=args.last_token_num)
test_encoded_last = testdata.map(process_fn)
test_encoded_last.set_format(type='torch', columns=['input_ids', 'attention_mask'])

training_args = TrainingArguments(
    output_dir="./results",
    per_device_eval_batch_size=16,
    report_to="none",  # 测试时禁用日志
    seed=42,
)

# 创建 Trainer（不需要 train_dataset）
trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
)

# 推理
predictions = trainer.predict(test_encoded_last)
preds = np.argmax(predictions.predictions, axis=-1)
print(model.config.id2label)
res={'A': 0, 'B': 0, 'C': 0, 'F': 0, 'X': 0}
for i in preds:
    if i == 0:
        res['A']+=1
    elif i==1:
        res['B']+=1
    elif i==2:
        res['C']+=1
    elif i==3:
        res['F']+=1
    elif i==4:
        res['X']+=1
#print(res)

id2label = model.config.id2label
bert_scored = []
with open(args.source_file, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        line = line.strip()
        if line:
            res = json.loads(line)
            res['A/B'] = id2label[preds[i]]
            bert_scored.append(res)


client = AsyncOpenAI(base_url=args.end_point, api_key=args.api_key)
semaphore = asyncio.Semaphore(100)

def _build_conversation_quality_judge_prompt(history: list[dict], language: str, assistant_goal: str) -> str:
    prompt = f"""销售对话整体评估
    1) 你的任务
    给你一段完整的销售对话，参与者为 assistant 与 user。请基于整段对话判断：assistant 是否对达成这笔交易产生了较大的贡献，并据此输出一个 0–10 的整数分数（仅输出数字）。

    assistant 的销售目标：{assistant_goal}

    完整对话：{history}

    2) 评估原则（只看对交易推进的实际影响）
    - 以 user 的回应与后续行动为准；未被接纳或未引发行动的不计正向贡献。
    - 只评估购买意向提升/交易推进，不评价文案或礼貌。
    - 采用保守判定；证据不足时取较低档位。
    - 若推进主要由 user 自发推动，应降低对 assistant 的贡献评分。

    3) 可计分的推进证据（示例，出现越多、越强，分数越高）
    - 口头同意购买（高权重）：user 明确口头承诺购买或下单（如“可以买”“就按这个下单”“我们决定采购”等），且未出现随后反悔。
    - 具体下一步被锁定：明确时间/参与人/议程并得到同意或已发起日程/邀请（演示、试点、技术评审、法务/采购流程节点等）。
    - 关键要素被获取并认可：预算、决策人、评估标准、时间表、合规要求等。
    - 异议被有效化解：重要顾虑被解除且不再阻碍下一步。
    - 承诺性动作：同意试用/POC、共享数据或接口、签署 NDA、推进采购/供应商注册、引入经济买家或技术评审。
    - 价值与场景匹配被坐实：需求—方案—收益链条得到 user 认可并与后续动作相呼应。

    4) 负向或不计分信号
    - 与交易无关、重复、被回避/拒绝、关键问题被搁置。
    - 仅 assistant 片面宣称价值而 user 未给出明确认可或行动。
    - 过度承诺、答非所问、造成误解，引发信任受损或推进受阻。
    - 承诺反悔：user 在口头同意后又撤回或否认，将显著降低评分。

    5) 评分标尺
    - 0–1 阻碍/倒退：对话使交易变差或被终止。
    - 2–3 无实质进展：未获关键信息，未锁定任何行动。
    - 4 轻微触动：出现意向迹象但无明确下一步或关键要素。
    - 5 有限推进：确认部分需求/价值；仅有模糊下一步或态度正面但未落地。
    - 6–7 清晰推进：出现一个具象推进点（关键要素被确认，或具体行动得到正面响应）。
    - 8–9 决定性推进：多个具象推进点或关键障碍被清除，促成里程碑式进展；或 user 明确口头同意购买但尚未触发流程。
    - 10 接近/达到成交：assistant 的引导直接促成口头同意购买并同步推进采购/合同签署/下单流程，或成交路径与时间表已明确并由 user 予以确认。

    6) 输出格式
    仅输出一个 0–10 的整数分数，除此之外不要输出任何文字、解释、单位或标点。


    你的打分输出：
    """
    request_msgs = [
        {"role": "user", "content": prompt},
    ]
    return request_msgs

async def judge_conversation_quality(history: list[dict], model_name: str, temperature: float, assistant_goal:str, reasoning_effort: str = "high", retry_count: int = 0)-> str:
    async with semaphore:
        request_msgs = _build_conversation_quality_judge_prompt(history, "chinese", assistant_goal)

        chat_completion = await client.chat.completions.create(
            messages=request_msgs,
            model=model_name,
            temperature=temperature,
            top_p=0.01,
            reasoning_effort=reasoning_effort,
        )
        res = chat_completion.choices[0].message.content or ""
        #think = getattr(chat_completion.choices[0].message, "reasoning_content", None)
        #match = re.search(r'(?s)(?:.*</think>\s*)?(.*)', res)
        return res
        #return (match.group(1) if match else res, think)

async def async_judge(current_msgs):
    model_name = args.llm_model_name
    temperature = 0.8
    assistant_goal = '成功向客户推销产品' 

    tasks = [judge_conversation_quality(json_msg['messages'], model_name, temperature, assistant_goal) for json_msg in current_msgs]
    results = await tqdm_asyncio.gather(*tasks)

    return results
       

res = asyncio.run(async_judge(bert_scored))
#print(res)


fname = args.source_file + '_scored.json'
label2score = {'A': 10, 'B': 8, 'C': 6, 'X': 4, 'F': 2}
report = []
with open(fname, "a", encoding="utf-8") as f:
    for i, line_scored in enumerate(bert_scored):
        line_scored['conversation_quality'] = int(res[i])
        line_scored['final_score'] = (1 - args.proportion) * line_scored['conversation_quality'] + args.proportion * label2score[line_scored['A/B']] #calculate the final scores
        report.append(line_scored['final_score'])
        json.dump(line_scored, f, ensure_ascii=False)
        f.write("\n")
        f.flush()

# print out the report
print('average score:', np.mean(report))
print('standard deviation:', np.std(report))
