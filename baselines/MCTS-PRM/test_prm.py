import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F


def make_step_rewards(logits, token_masks):
    probabilities = F.softmax(logits, dim=-1)
    probabilities = probabilities * token_masks.unsqueeze(-1) # bs, seq_len, num_labels
    
    all_scores_res = []
    for i in range(probabilities.size(0)):
        sample = probabilities[i] # seq_len, num_labels
        positive_probs = sample[sample != 0].view(-1, 2)[:, 1] # valid_tokens, num_labels
        non_zero_elements_list = positive_probs.cpu().tolist()
        all_scores_res.append(non_zero_elements_list)
    return all_scores_res


model_name = "/home2/kwshi/model/Qwen2.5-Math-PRM-7B"
device = "auto"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name, 
    device_map=device, 
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).eval()


data = {
    "system": "Please reason step by step, and put your final answer within \\boxed{}.",
    "query": "How many prime numbers are between 20 and 30?",
    "response": ["Step 3: Count the number of primes found.\n\nThere are 2 prime numbers between 20 and 30.\n\nFinal answer:\\boxed{2}",
                 "3: Count the number of primes.\nOnly 23 is prime.\nFinal answer:\\boxed{1}"
                ]
}

messages = [
    {"role": "system", "content": data['system']},
    {"role": "user", "content": data['query']},
    {"role": "assistant", "content": "<extra_0>".join(data['response']) + "<extra_0>"},
]
conversation_str = tokenizer.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=False
)

input_ids = tokenizer.encode(
    conversation_str, 
    return_tensors="pt", 
).to(model.device)

outputs = model(input_ids=input_ids)

good = ["Step 1: Identify all numbers between 20 and 30.\n\n", "Step 2: For each number, check if it is prime.\n- 20: even, not prime\n- 21: divisible by 3, not prime\n- 22: even, not prime\n- 23: prime\n- 24: even, not prime\n- 25: divisible by 5, not prime\n- 26: even, not prime\n- 27: divisible by 3, not prime\n- 28: even, not prime\n- 29: prime\n- 30: even, not prime\n\n", "Step 3: Count the number of primes found.\n\nThere are 2 prime numbers between 20 and 30.\n\nFinal answer:\\boxed{2}",]

step_sep_id = tokenizer.encode("<extra_0>")[0]
token_masks = (input_ids == step_sep_id)
step_reward = make_step_rewards(outputs[0], token_masks)
print(step_reward)  # [[1.0, 0.1904296875, 0.9765625, 1.0]]