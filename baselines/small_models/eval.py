import argparse
import json
import os
import re
from dataclasses import field, dataclass

import torch
from sympy.parsing.sympy_parser import parse_expr
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedModel, HfArgumentParser, AutoModelForCausalLM, GenerationConfig, PreTrainedTokenizer, MBartForConditionalGeneration, \
    MT5ForConditionalGeneration

from dataset import DatasetFormulaLM


@dataclass
class ModelArguments:
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
                            pad_token_id=tokenizer.pad_token_id)
    outputs = tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return outputs


def eval_qwen(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, dataset):
    pred_f = []
    for data_idx in tqdm(range(len(dataset)), total=len(dataset), desc='evaluating...'):
        feature = dataset[data_idx]
        input_ids = torch.tensor(feature['input_ids'], device=model.device).unsqueeze(0)
        attention_mask = torch.tensor(feature['attention_mask'], device=model.device).unsqueeze(0)
        outputs = genearte_impl(input_ids, attention_mask, model, tokenizer)
        prompt = tokenizer.decode(input_ids[0])
        question = re.findall(r'Question:(.*?)Answer:', prompt)[0].strip() if 'Question:' in prompt else prompt.split('\n\n')[-1].strip()
        this_pred_f = []
        for output in outputs:
            try:
                for per in re.findall(r'\d+%', output):
                    output = output.replace(per, f'_{per[:-1]}_PERCENT_')
                output = output.replace('℃', '摄氏度')
                formulas = re.split(r'[,，]', output)
                formulas = list(map(str.strip, formulas))
                formulas = list(map(lambda x: x.replace(' ', ''), formulas))
                formulas = list(map(lambda x: x.replace('^', '**'), formulas))
                formulas = list(map(lambda x: x.replace('{', ''), formulas))
                formulas = list(map(lambda x: x.replace('}', ''), formulas))
                formulas = list(map(lambda x: x.split('=', 1), formulas))
                left_params = list(map(lambda x: x[0].strip(), formulas))
                right = list(map(lambda x: x[1].strip(), formulas))
                right = list(map(parse_expr, right))
                right_params = sum(list(map(lambda expr: list(map(str, list(expr.free_symbols))), right)), [])
                right_params = list(filter(lambda x: x not in left_params, right_params))

                para_prompts = [f'这是一个初中物理题目，找出对应概念的数值和单位。\n\nQuestion: {question}\n概念: {p}\n数值: ' for p in right_params]
                para_prompts_inputs = tokenizer.__call__(para_prompts, max_length=256, return_tensors='pt', padding=True)
                para_prompts_input_ids = para_prompts_inputs.input_ids.to(model.device)
                para_prompts_attention_mask = para_prompts_inputs.attention_mask.to(model.device)
                para_outputs = [genearte_impl(para_prompts_input_ids[idx].unsqueeze(0),
                                              model=model,
                                              attention_mask=para_prompts_attention_mask[idx].unsqueeze(0),
                                              tokenizer=tokenizer,
                                              num_return_sequences=1,
                                              do_sample=False)[0] for idx in range(para_prompts_input_ids.size(0))]

                this_pred_f.append({'question': question, 'formula': output, 'symbol_map': dict(zip(right_params, para_outputs))})
            except Exception as e:
                print(e)
                print(f'failed to parse {output}')
                this_pred_f.append(None)
        pred_f.append({'id': feature['id'], 'preds': this_pred_f, 'original_data': feature['original_data']})
    return pred_f


def main():
    parser = HfArgumentParser((ModelArguments,))
    model_args, = parser.parse_args_into_dataclasses()

    if 'mt5' in model_args.model_name_or_path.lower():
        model_class = MT5ForConditionalGeneration
        model_size = 'large' if 'large' in model_args.model_name_or_path.lower() else 'base'
    else:
        raise NotImplementedError(f'unrecognized model type form {model_args.model_name_or_path}')

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model = model_class.from_pretrained(model_args.model_name_or_path,
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
