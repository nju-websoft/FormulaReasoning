from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import LoraConfig, get_peft_model
from trl import DPOConfig, DPOTrainer
import torch
# from swanlab.integration.transformers import SwanLabCallback
import sys
import os
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from math_utils import clean_expr_str
from grader import math_equal
from transformers import set_seed
from data_collator import DataCollatorForPreference

set_seed(1234)

# swanlab_callback = SwanLabCallback(
#     project="DeepSeek-R1-Distill-Qwen-1.5B-dpo",
#     experiment_name="DeepSeek-R1-Distill-Qwen-1.5B",
#     description="进行gsm8k+math with Q的step dpo",
#     config={
#         "model": "DeepSeek-R1-Distill-Qwen-1.5B-dpo",
#         "dataset": "gsm8k+math preference_data",
#     }
# )

dataset = load_dataset("json", data_files={"train": "../MCTS-PRM/output_formulareasoning.jsonl"})["train"]


# dataset2 = load_dataset("json", data_files={"train": "/home2/kwshi/skw/MCTS-GSM8K/output_MATH_2500_3000_with_Q.jsonl"})
# dataset3 = load_dataset("json", data_files={"train": "/home2/kwshi/skw/MCTS-GSM8K/output_MATH_3000_3500_with_Q.jsonl"})
# dataset4 = load_dataset("json", data_files={"train": "/home2/kwshi/skw/MCTS-GSM8K/output_MATH_3500_4000_with_Q.jsonl"})
# dataset = concatenate_datasets([dataset1["train"], dataset2["train"], dataset3["train"], dataset4["train"]])


def extract_pred_answer(pred_str):
    pred = ""
    if "boxed" in pred_str:
        ans = pred_str.split("boxed")[-1]
        if len(ans) == 0:
            return ""
        elif ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
        pred = a
    return pred


def formula_filter(sample, best_has_formula=True, worst_has_formula=True):
    best_steps = sample["steps"]["best"]
    worst_steps = sample["steps"]["worst"]

    formula_pattern = re.compile(r'\[.*?\]')

    def all_steps_have_formulas(steps):
        for step in steps:
            if not re.search(formula_pattern, step):
                return False
        return True

    def all_steps_not_have_formulas(steps):
        for step in steps:
            if re.search(formula_pattern, step):
                return False
        return True

    if best_has_formula:
    #    if not all_steps_have_formulas(best_steps):
    #        return False
    #else:
        if all_steps_not_have_formulas(best_steps):
            return False
    if worst_has_formula:
    #    if not all_steps_have_formulas(worst_steps):
    #        return False
    #else:
        if all_steps_not_have_formulas(worst_steps):
            return False
    return True


def preprocess_filter(sample):
    # 检查步骤有效性
    best_steps = sample["steps"]["best"]
    worst_steps = sample["steps"]["worst"]

    if len(best_steps) == 0 or len(worst_steps) == 0:
        print(sample["Question"])
        return False

    if "Q" in sample:  # using Q/N to get reward
        # 检查Q/N数据完整性
        q_best = sample["Q"]["best"]
        n_best = sample["N"]["best"]
        q_worst = sample["Q"]["worst"]
        n_worst = sample["N"]["worst"]
        if len(q_best) != len(best_steps) or len(n_best) != len(best_steps):
            return False
        if len(q_worst) != len(worst_steps) or len(n_worst) != len(worst_steps):
            return False
    else:  # using process supervised verifier to get reward
        if len(sample["Step_reward"]["best"]) != len(best_steps) or len(sample["Step_reward"]["worst"]) != len(
                worst_steps):
            print(sample["Question"])
            return False
    # 检查答案有效性
    chosen_ans_raw = best_steps[-1]
    rejected_ans_raw = worst_steps[-1]
    chosen_ans = clean_expr_str(extract_pred_answer(chosen_ans_raw))
    rejected_ans = clean_expr_str(extract_pred_answer(rejected_ans_raw))
    ground_truth = clean_expr_str(sample["ground_truth"])

    # 验证答案逻辑
    #if not math_equal(chosen_ans, ground_truth):
    #    return False
    #if math_equal(rejected_ans, ground_truth):
    #    return False
    #if math_equal(chosen_ans, rejected_ans):
    #    return False

    # 检查响应长度
    chosen_response = "\n".join(best_steps)
    rejected_response = "\n".join(worst_steps)
    if len(chosen_response) > 3072 or len(rejected_response) > 3072:
        return False

    return True


def process_data(sample):
    # 计算每个步骤的奖励（Q/N）
    def calculate_rewards(q_list, n_list):
        return [q / (n + 1e-6) for q, n in zip(q_list, n_list)]  # 防止除零

    if "Q" in sample:  # using Q/N to get reward
        best_rewards = calculate_rewards(sample["Q"]["best"], sample["N"]["best"])
        worst_rewards = calculate_rewards(sample["Q"]["worst"], sample["N"]["worst"])
    else:  # using process supervised verifier to get reward
        best_rewards = sample["Step_reward"]["best"]
        worst_rewards = sample["Step_reward"]["worst"]
    return {
        "prompt": sample["Question"],
        "chosen": "\n".join(sample["steps"]["best"]),
        "rejected": "\n".join(sample["steps"]["worst"]),
        "best_rewards": best_rewards,
        "worst_rewards": worst_rewards,
        "best_steps": sample["steps"]["best"],
        "worst_steps": sample["steps"]["worst"]
    }


filtered_dataset = dataset.filter(preprocess_filter)
filtered_dataset = filtered_dataset.filter(formula_filter)
train_dataset = filtered_dataset.map(process_data)

name = 'DeepSeek-R1-Distill-Qwen-7B'
#name = 'Qwen2.5-Math-1.5B-Instruct'
#name = 'Qwen2___5-1___5B-Instruct'
model_name = f"/home/blzhu/Qwen2.5-Math/models/{name}"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# Apply LoRA configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ]
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Define training arguments
training_args = DPOConfig(
    output_dir=f'./output_fr/{name}/dpo',
    beta=0.1,
    learning_rate=5e-7,
    max_length=2048,
    warmup_ratio=0.05,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=30,
    save_total_limit=1,
    logging_dir='./logs',
    logging_steps=200,
    save_strategy="epoch",
    remove_unused_columns=False,
    report_to=[]
)

dpo_data_collator = DataCollatorForPreference(pad_token_id=tokenizer.pad_token_id)

dpo_trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer,
    # callbacks=[swanlab_callback],
    data_collator=dpo_data_collator
)

dpo_trainer.train()

model = model.merge_and_unload()
save_model_path = f"./model_fr/{name}/dpo"

model.save_pretrained(
    save_model_path,
    safe_serialization=True,
)
tokenizer.save_pretrained(save_model_path)
