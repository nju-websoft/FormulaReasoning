import argparse
import os.path
from dataclasses import field, dataclass

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, Trainer, HfArgumentParser, TrainingArguments, AutoModelForCausalLM, DataCollatorWithPadding, MT5ForConditionalGeneration, \
    MBartForConditionalGeneration


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default='/path/to/mt5-base',
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )


@dataclass
class DataTrainingArguments:
    task_name: str = field(default='formula_lm',
                           metadata={"help": "The name of the task to train"})
    data_dir: str = field(default='../../data/dataset_v2/',
                          metadata={"help": "Should contain the data files for the task."})
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if 'mbart' in model_args.model_name_or_path.lower():
        model_class = MBartForConditionalGeneration
    elif 'mt5' in model_args.model_name_or_path.lower():
        model_class = MT5ForConditionalGeneration
    else:
        raise NotImplementedError(f'unrecognized model type form {model_args.model_name_or_path}')

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    model = model_class.from_pretrained(model_args.model_name_or_path,
                                        device_map="auto",
                                        trust_remote_code=True, )

    def tokenize(examples):
        if 'instruction' in examples:
            input_strs = [f'{inst}\n\n{input}' for inst, input in zip(examples.pop('instruction'), examples.pop('input'))]
        else:
            input_strs = examples.pop('prompt')
        inputs = tokenizer.__call__(input_strs, truncation=True, max_length=data_args.max_seq_length)
        outputs = tokenizer.__call__(examples.pop('output' if 'output' in examples else 'program'), truncation=True, max_length=data_args.max_seq_length)

        return {'input_ids': inputs.input_ids, 'attention_mask': inputs.attention_mask, 'labels': outputs.input_ids}

    train_dataset = load_dataset('json', data_files=os.path.join(data_args.data_dir, 'train.json'))['train']
    train_dataset = train_dataset.map(tokenize, batched=True, num_proc=16)

    def data_collator_fn(features):
        batch_max_length = max([len(e['input_ids']) for e in features])
        batch_max_length_labels = max([len(e['labels']) for e in features])

        input_ids = torch.tensor([f['input_ids'] + [tokenizer.pad_token_id] * (batch_max_length - len(f['input_ids'])) for f in features])
        attention_mask = torch.tensor([f['attention_mask'] + [0] * (batch_max_length - len(f['attention_mask'])) for f in features])
        labels = torch.tensor([f['labels'] + [-100] * (batch_max_length_labels - len(f['labels'])) for f in features])

        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=train_dataset,
                      data_collator=data_collator_fn)

    trainer.train()
    trainer.save_model()


if __name__ == '__main__':
    main()
