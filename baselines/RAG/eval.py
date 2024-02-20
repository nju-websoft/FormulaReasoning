import argparse
import json
import os.path
import random

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from retriever import FormulaRetriever

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--max_seq_length', type=int, default=256)
parser.add_argument('--data_dir', type=str, default='../../data/dataset_formula_retriever')
args = parser.parse_args()


def eval(retriever: FormulaRetriever, dataloader: DataLoader):
    retriever.eval()
    results = {}
    hit = {1: set(), 5: set(), 10: set(), 20: set()}
    recall = {1: [], 5: [], 10: [], 20: []}
    for batch in tqdm(dataloader):
        ids = batch.pop('ids')
        labels = batch.pop('labels')
        with torch.no_grad():
            scores = retriever.pred(**batch)['scores'].detach().cpu().numpy().tolist()

        for id, label, score in zip(ids, labels, scores):
            score_rank = sorted(zip(range(retriever.n_formula), score), key=lambda x: float(x[1]), reverse=True)
            for idx in recall.keys():
                sort_rank_ids = list(map(lambda x: x[0], score_rank))
                if len(label) == 0:
                    recall[idx].append(0)
                else:
                    recall[idx].append(len(list(filter(lambda x: x in label, sort_rank_ids))) / float(len(label)))

        for id, label, score in zip(ids, labels, scores):
            score_rank = sorted(zip(range(retriever.n_formula), score), key=lambda x: float(x[1]), reverse=True)
            for idx in range(max(hit.keys())):
                if score_rank[idx][0] in label:
                    for key in hit:
                        if idx < key:
                            hit[key].add(id)
            results[id] = {'label': label, 'top_20': [list(d) for d in score_rank[:20]]}
    for key in hit:
        hit[key] = len(hit[key]) / (len(dataloader) * args.batch_size)
    for key in recall:
        recall[key] = sum(recall[key]) / len(recall[key])

    all_formulas = json.load(open('../../data/raw/formulas.json'))
    for key in results:
        for idx in range(len(results[key]['label'])):
            results[key]['label'][idx] = all_formulas[results[key]['label'][idx]]
        for idx in range(len(results[key]['top_20'])):
            results[key]['top_20'][idx] = all_formulas[results[key]['top_20'][idx][0]]

    return results, hit, recall


def main():
    ckp = torch.load(os.path.join(args.model_path, 'pytorch_model.bin'))
    retriever = FormulaRetriever(model_path='/data1/PTLM/chinese_wwm_bert_ext/')
    retriever.load_state_dict(ckp)
    device = 'cuda:0'
    retriever.to(device)
    tokenizer = AutoTokenizer.from_pretrained('/data1/PTLM/chinese_wwm_bert_ext/')

    id_test = load(tokenizer, 'id_test')
    ood_test = load(tokenizer, 'ood_test')

    def data_collator_fn(features):
        batch_max_length = max([len(e['input_ids']) for e in features])
        input_ids, attention_mask, labels, ids = [], [], [], []

        for f in features:
            id = f['q_id']
            ids.append(id)
            input_ids.append(f['input_ids'] + [tokenizer.pad_token_id] * (batch_max_length - len(f['input_ids'])))
            attention_mask.append(f['attention_mask'] + [0] * (batch_max_length - len(f['attention_mask'])))
            labels.append(f['label'])

        input_ids = torch.tensor(input_ids, device=device)
        attention_mask = torch.tensor(attention_mask, device=device)

        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels, 'ids': ids}

    id_test_dataloader = DataLoader(dataset=id_test,
                                    batch_size=args.batch_size,
                                    collate_fn=data_collator_fn,
                                    pin_memory=False)
    ood_test_dataloader = DataLoader(dataset=ood_test,
                                     batch_size=args.batch_size,
                                     collate_fn=data_collator_fn,
                                     pin_memory=False)

    print(f'eval id test...')
    id_results, id_hit, id_recall = eval(retriever, id_test_dataloader)
    print(f'{id_hit=}')
    print(f'{id_recall=}')
    with open('id_results.json', 'w', encoding='utf-8') as f:
        json.dump(id_results, f, ensure_ascii=False, indent=4)

    print(f'eval ood test...')
    ood_results, ood_hit, ood_recall = eval(retriever, ood_test_dataloader)
    print(f'{ood_hit=}')
    print(f'{ood_recall=}')
    with open('ood_results.json', 'w', encoding='utf-8') as f:
        json.dump(ood_results, f, ensure_ascii=False, indent=4)


def load(tokenizer, data_type):
    def tokenize(examples):
        inputs = tokenizer.__call__(examples.pop('input'), truncation=True, max_length=args.max_seq_length)
        return {'input_ids': inputs.input_ids, 'attention_mask': inputs.attention_mask, 'label': examples.pop('label'), 'q_id': examples.pop('id')}

    dataset = load_dataset('json', data_files=os.path.join(args.data_dir, f'{data_type}.json'))['train']
    dataset = dataset.map(tokenize, batched=True, num_proc=16)
    return dataset


if __name__ == '__main__':
    main()
