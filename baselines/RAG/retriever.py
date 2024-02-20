import argparse
import json
import os
import random
from dataclasses import dataclass, field
from io import StringIO
from typing import List

import torch
from datasets import load_dataset
from torch import nn
from transformers import BertModel, AutoModel, AutoConfig, AutoTokenizer, HfArgumentParser, TrainingArguments, Trainer
import torch.nn.functional as F


@dataclass
class ModelArguments:
    model_path: str = field(
        default='/path/to/chinese_wwm_bert_ext/',
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    max_seq_length: int = field(
        default=256
    )
    data_dir: str = field(
        default='../../data/dataset_formula_retriever',
    )


class FormulaRetriever(nn.Module):
    def __init__(self, model_path, n_formula=272, margin=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = AutoModel.from_pretrained(model_path)
        self.config = AutoConfig.from_pretrained(model_path)
        self.formula_embs = nn.Embedding(n_formula, self.config.hidden_size, padding_idx=-1)
        self.margin = margin  # Margin for contrastive loss
        self.tau = 1.0
        self.n_formula = n_formula

    def forward(self, input_ids, attention_mask, labels):
        input_embs = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state
        text_embs = input_embs[:, 0, :]
        formula_embs = self.formula_embs(labels)
        text_embs = F.normalize(text_embs, p=2, dim=1)
        formula_embs = F.normalize(formula_embs, p=2, dim=1)

        sim_scores = torch.matmul(text_embs, formula_embs.t()) / self.tau
        targets = torch.arange(sim_scores.size(0)).to(sim_scores.device)

        loss = nn.CrossEntropyLoss()(sim_scores, targets)

        return {'loss': loss}

    def pred(self, input_ids, attention_mask):
        input_embs = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state
        text_embs = input_embs[:, 0, :]
        formula_embs = self.formula_embs(torch.arange(self.n_formula).to(self.encoder.device))

        text_embs = F.normalize(text_embs, p=2, dim=1)
        formula_embs = F.normalize(formula_embs, p=2, dim=1)

        sim_scores = torch.matmul(text_embs, formula_embs.t())
        return {'scores': sim_scores}


def main():
    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses()

    retriever = FormulaRetriever(args.model_path)
    device = 'cuda:0'
    retriever.to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    def tokenize(examples):
        inputs = tokenizer.__call__(examples.pop('input'), truncation=True, max_length=args.max_seq_length)
        return {'input_ids': inputs.input_ids, 'attention_mask': inputs.attention_mask, 'label': examples.pop('label'), 'q_id': examples.pop('id')}

    raw_datas = json.load(open(os.path.join(args.data_dir, 'train.json')))
    datas = []
    for line in raw_datas:
        for formula_id in line['label']:
            datas.append({'input': line['input'], 'label': formula_id, 'id': line['id']})
    random.shuffle(datas)
    with open('.__tmp_cache_file__', 'w', encoding='utf-8') as f:
        json.dump(datas, f, ensure_ascii=False)
    train_dataset = load_dataset('json', data_files='.__tmp_cache_file__', )['train']
    train_dataset = train_dataset.map(tokenize, batched=True, num_proc=16)

    def data_collator_fn(features):
        batch_max_length = max([len(e['input_ids']) for e in features])
        used_ids = set()
        input_ids, attention_mask, labels = [], [], []

        for f in features:
            id = f['q_id']
            if id in used_ids:
                continue
            used_ids.add(id)
            input_ids.append(f['input_ids'] + [tokenizer.pad_token_id] * (batch_max_length - len(f['input_ids'])))
            attention_mask.append(f['attention_mask'] + [0] * (batch_max_length - len(f['attention_mask'])))
            labels.append(f['label'])

        input_ids = torch.tensor(input_ids, device=device)
        attention_mask = torch.tensor(attention_mask, device=device)
        labels = torch.tensor(labels, device=device)

        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

    trainer = Trainer(model=retriever,
                      args=training_args,
                      train_dataset=train_dataset,
                      data_collator=data_collator_fn)

    trainer.train()
    trainer.save_model()


if __name__ == '__main__':
    main()
