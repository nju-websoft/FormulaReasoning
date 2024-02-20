import argparse
import json
import os
import re
from dataclasses import field, dataclass

import torch
from sympy.parsing.sympy_parser import parse_expr
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedModel, HfArgumentParser, AutoModelForCausalLM, GenerationConfig, PreTrainedTokenizer

from utils.dataset import DatasetFormulaLM


@dataclass
class ModelArguments:
    # n_formula: int = field(metadata={'help': 'number of formulas'})
    # formula_token: str = field(metadata={'help': 'formula token'})
    model_name_or_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    max_seq_length: int = field(default=512)
    data_dir: str = field(default='../../data/dataset_v2/')
    data_type: str = field(default='id_test')
    eval_name: str = field(default='')


def genearte_impl(input_ids, attention_mask, model, tokenizer, num_return_sequences=5, do_sample=True):
    output = model.generate(input_ids=input_ids,
                            attention_mask=attention_mask,
                            return_dict_in_generate=False,
                            generation_config=GenerationConfig(max_new_tokens=128,
                                                               do_sample=do_sample,
                                                               top_p=0.8,
                                                               top_k=0,
                                                               num_return_sequences=num_return_sequences),
                            eos_token_id=tokenizer.eos_token_id,
                            pad_token_id=tokenizer.pad_token_id)[:, input_ids.size(-1):]
    outputs = tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return outputs


def eval_qwen(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, dataset):
    pred_f = []
    for data_idx in tqdm(range(len(dataset)), total=len(dataset), desc='evaluating...'):
        feature = dataset[data_idx]
        input_ids = torch.tensor(feature['input_ids'], device=model.device).unsqueeze(0)
        attention_mask = torch.tensor(feature['attention_mask'], device=model.device).unsqueeze(0)
        outputs = genearte_impl(input_ids, attention_mask, model, tokenizer)
        pred_f.append({'id': feature['id'], 'preds': outputs, 'original_data': feature['original_data']})
    return pred_f


def main():
    parser = HfArgumentParser((ModelArguments,))
    model_args, = parser.parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path,
                                              pad_token='<|extra_0|>',
                                              eos_token='<|endoftext|>',
                                              padding_side='left',
                                              truncation_side='left',
                                              trust_remote_code=True, )

    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path,
                                                 pad_token_id=tokenizer.pad_token_id,
                                                 device_map="auto",
                                                 trust_remote_code=True, )

    if model_args.data_type == 'id_test':
        id_test_dataset = DatasetFormulaLM(tokenizer=tokenizer,
                                           max_seq_length=model_args.max_seq_length,
                                           data_dir=model_args.data_dir,
                                           data_type='id_test')
        id_results = eval_qwen(model, tokenizer=tokenizer, dataset=id_test_dataset)
        with open(f'id_results_{model_args.eval_name}.json', 'w', encoding='utf-8') as f:
            json.dump(id_results, f, ensure_ascii=False, indent=4)

    if model_args.data_type == 'ood_test':
        ood_test_dataset = DatasetFormulaLM(tokenizer=tokenizer,
                                            max_seq_length=model_args.max_seq_length,
                                            data_dir=model_args.data_dir,
                                            data_type='ood_test')
        ood_results = eval_qwen(model, tokenizer=tokenizer, dataset=ood_test_dataset)
        with open(f'ood_results_{model_args.eval_name}.json', 'w', encoding='utf-8') as f:
            json.dump(ood_results, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()
