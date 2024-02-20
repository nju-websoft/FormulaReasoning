import json
import math
from dataclasses import field, dataclass

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, TrainingArguments, AutoModelForCausalLM, GenerationConfig, PreTrainedModel, PreTrainedTokenizer, AutoModel


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default='/path/to/local/llm',
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )


@dataclass
class DataTrainingArguments:
    data_file: str = field(default='datas/id_test_zero_shot.json',
                           metadata={"help": "Should contain the data files for the task."})
    max_seq_length: int = field(
        default=2048,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    batch_size: int = field(default=2)
    task_name: str = field(default='results_llama2_7b')


def prepare_model(model_name_or_path):
    if 'qwen' in model_name_or_path.lower():
        print(f'load Qwen model from {model_name_or_path}')
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                                  pad_token='<|extra_0|>',
                                                  eos_token='<|endoftext|>',
                                                  padding_side='left',
                                                  truncation_side='left',
                                                  trust_remote_code=True, )

        model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                     pad_token_id=tokenizer.pad_token_id,
                                                     device_map="auto",
                                                     trust_remote_code=True, )
    elif 'internlm' in model_name_or_path.lower() or 'llama' in model_name_or_path.lower():
        print(f'load {"internlm" if "internlm" in model_name_or_path.lower() else "llama"} model from {model_name_or_path}')
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                                  padding_side='left',
                                                  truncation_side='left', )
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise NotImplementedError(f'both eos token and pad token is None')
            else:
                tokenizer.pad_token_id = tokenizer.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                     device_map="auto",
                                                     pad_token_id=tokenizer.pad_token_id,
                                                     trust_remote_code=True,
                                                     torch_dtype=torch.float16)
        tokenizer.padding_side = 'left'
        tokenizer.truncation_side = 'left'
        print(f'{tokenizer.pad_token=}')
        print(f'{tokenizer.eos_token=}')
    elif 'chatglm' in model_name_or_path.lower():
        print(f'load ChatGLM model from {model_name_or_path}')
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name_or_path,
                                          trust_remote_code=True,
                                          pad_token_id=tokenizer.pad_token_id,
                                          device_map="auto", )
    else:
        raise NotImplementedError(f'unrecognized model from {model_name_or_path}')
    model.eval()
    return tokenizer, model


def generate_impl(input_ids, attention_mask, model, tokenizer, do_sample=True):
    output = model.generate(input_ids=input_ids,
                            attention_mask=attention_mask,
                            return_dict_in_generate=False,
                            generation_config=GenerationConfig(max_new_tokens=512,
                                                               eos_token_id=tokenizer.eos_token_id,
                                                               do_sample=do_sample,
                                                               top_p=0.8, ))[:, input_ids.size(-1):]
    outputs = tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return outputs


def eval_llm(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, dataset, data_collator_fn, batch_size):
    llm_outputs = {}
    for data_idx in tqdm(range(0, len(dataset), batch_size), total=math.ceil(len(dataset) / batch_size), desc='evaluating...'):
        features = dataset[data_idx:data_idx + batch_size]
        batch = data_collator_fn(features, model.device)
        outputs = generate_impl(**batch, model=model, tokenizer=tokenizer)
        ids = features['id']
        llm_outputs |= dict(zip(ids, outputs))
    return llm_outputs


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

    tokenizer, model = prepare_model(model_args.model_name_or_path)

    def tokenize(examples):
        inputs = tokenizer.__call__(examples['prompt'], truncation=True, max_length=data_args.max_seq_length)

        return {'input_ids': inputs.input_ids, 'attention_mask': inputs.attention_mask}

    eval_dataset = load_dataset('json', data_files=data_args.data_file, split='train')
    eval_dataset = eval_dataset.map(tokenize, batched=True, num_proc=16)

    def data_collator_fn(features, device):
        batch_max_length = max(map(len, features['input_ids']))

        # padding side = left
        input_ids = torch.tensor([[tokenizer.pad_token_id] * (batch_max_length - len(f)) + f for f in features['input_ids']], device=device)
        attention_mask = torch.tensor([[0] * (batch_max_length - len(f)) + f for f in features['attention_mask']], device=device)
        return {'input_ids': input_ids, 'attention_mask': attention_mask}

    llm_outputs = eval_llm(model=model,
                           tokenizer=tokenizer,
                           dataset=eval_dataset,
                           data_collator_fn=data_collator_fn,
                           batch_size=data_args.batch_size)

    with open(data_args.data_file.replace('.json', f'.{data_args.task_name}.json'), 'w', encoding='utf-8') as f:
        json.dump(llm_outputs, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()
