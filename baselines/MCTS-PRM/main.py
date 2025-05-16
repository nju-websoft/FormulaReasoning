import json
import logging
import os

# custom imports
from mcts_tree import MCTS
from node import QuestionNode
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
from data_loader import load_data
from vllm import LLM


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

num_iterations = 24
num_children = 4
data_dir = "./data/MetaMathQA"
# data_dir = "/home/xli/skw/MCTS-GSM8k-Demo/qa_test"
qa_pairs = load_data(data_dir, "train")

device = "auto"
model_path = "/home2/blzhu/DeepSeek-R1-Distill-Qwen-14B"
prm_model_name = "/home2/kwshi/model/Qwen2.5-Math-PRM-7B"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = LLM(model=model_path,tensor_parallel_size=4,max_model_len=2048)
# model = AutoModelForCausalLM.from_pretrained(
#      model_path,
#      torch_dtype=torch.bfloat16,
#      device_map=device,
#      trust_remote_code=True,
#      ).eval()
prm_tokenizer = AutoTokenizer.from_pretrained(prm_model_name, trust_remote_code=True)
prm_model = AutoModel.from_pretrained(
    prm_model_name, 
    device_map=device, 
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).eval()

mcts = MCTS()

# 选择平均Reward最高/最低的路径
def traverse_path(node, is_best_path, mcts):
    """遍历所有可能的路径，并根据平均Reward返回最佳或最差路径"""
    all_paths = []
    
    def helper(current_node, current_steps, current_rewards):
        # 添加当前节点的子节点到路径中
        if current_node.is_terminal() or current_node not in mcts.children:
            all_paths.append((current_steps.copy(), current_rewards.copy()))
            return
        # 遍历所有子节点，递归生成路径
        for child in mcts.children[current_node]:
            new_steps = current_steps + [str(child.output)]
            new_rewards = current_rewards + [mcts.Reward[child]]
            helper(child, new_steps, new_rewards)
    
    # 初始化路径收集
    if not node.is_terminal() and node in mcts.children:
        for child in mcts.children[node]:
            helper(child, [str(child.output)], [mcts.Reward[child]])
    
    # 计算每条路径的平均Reward
    if not all_paths:
        return [], []  # 没有路径时返回空
    
    path_avg = []
    for steps, rewards in all_paths:
        avg = sum(rewards) / len(rewards) if rewards else 0
        path_avg.append((avg, steps, rewards))
    
    # 根据is_best_path选择最佳或最差路径
    if is_best_path:
        selected = max(path_avg, key=lambda x: x[0])
    else:
        selected = min(path_avg, key=lambda x: x[0])
    
    return selected[1], selected[2]

with open('output_MetaMathQA_0_5000.jsonl','a',encoding='UTF-8') as json_file:
    for qa_pair in tqdm(qa_pairs[:5000], desc="Processing questions"):
        root_node = QuestionNode(qa_pair["question"], model, tokenizer, prm_model, prm_tokenizer, None, num_children=num_children)
        
        for i in tqdm(range(num_iterations)):
            mcts.do_iteration(root_node)
        
        best_steps, best_step_reward = traverse_path(root_node, True, mcts)
        worst_steps, worst_step_reward = traverse_path(root_node, False, mcts)
    
        output_data = {
            "Question": qa_pair["question"],
            "ground_truth": qa_pair["ground_truth"],
            "steps": {
                "best": best_steps,
                "worst": worst_steps
            },
            "Step_reward":{
                "best": best_step_reward,
                "worst": worst_step_reward
            }
        }

        json_line = json.dumps(output_data, ensure_ascii=False)
        json_file.write(json_line + '\n')
        json_file.flush()
        # mcts._visualize(f"mcts_tree_{qa_pair['idx']}") 

