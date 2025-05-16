import torch
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

def get_step_reward(self):
    messages = [
        {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
        {"role": "user", "content": self.question},
        {"role": "assistant", "content": self.output + "<extra_0>"},
    ]
    conversation_str = self.prm_tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=False
    )
    input_ids = self.prm_tokenizer.encode(
        conversation_str, 
        return_tensors="pt", 
    ).to(self.prm_model.device)
    outputs = self.prm_model(input_ids=input_ids)

    step_sep_id = self.prm_tokenizer.encode("<extra_0>")[0]
    token_masks = (input_ids == step_sep_id)
    step_reward = make_step_rewards(outputs[0], token_masks)
    return step_reward