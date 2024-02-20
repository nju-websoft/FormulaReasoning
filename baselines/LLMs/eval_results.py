import argparse
import json
import os.path
import re
import subprocess

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--id_results', type=str, required=True)
parser.add_argument('--ood_results', type=str, default=None)
args = parser.parse_args()

zh_unit_map = {'立方米': 'm³', '度电': 'kWh', '千瓦': 'kW', '瓦': 'W', '千克': 'kg', '克': 'g', '\\times': '*', '\\text': '', '焦耳': 'J', '帕斯卡': 'Pa', '帕': 'Pa',
               '^\\circ C': '℃', '米': 'm', '千米': 'km', '公里': 'km', '伏': 'V', '千伏': 'kV', '千瓦时': 'kWh', '{': '', '}': '', '\\': '', '欧姆': 'Ω', '安培': 'A',
               '安': 'A', '℃': 'K', '分钟': 'min', '百分比': '%', '°C': 'K', '秒': 's', '小时': 'h', ',': '', '，': ''}


def extract_answer(s):
    if isinstance(s, dict):
        return None
    s = s.replace('\n单位：', '')
    s = s.split('###')[-1]
    s = s.split('Question:')[0]
    s = s.split('最终答案是：')[-1]
    s = s.split('最终答案：')[-1]
    s = s.split('最终答案是:')[-1]
    s = s.split('最终答案:')[-1]
    s = s.split('最终答案是')[-1]
    s = s.split('最终答案')[-1].strip()

    for zh_unit in sorted(list(zh_unit_map.keys()), key=len, reverse=True):
        s = s.replace(zh_unit, zh_unit_map[zh_unit])

    s = s.split('\n')
    s = list(filter(lambda x: x.strip() != '', s))
    s = list(map(lambda x: re.findall(r'\d+[\da-zA-Z ^/%³.\\×x℃{}Ω*\-]+', x), s))
    s = list(filter(lambda x: len(x) != 0, s))
    if len(s) == 0:
        return None
    else:
        s = s[-1]

    if len(s) == 0:
        return None

    s = s[-1]  # max(s, key=len)
    return s


def process_gold_answer(s):
    for zh_unit in sorted(list(zh_unit_map.keys()), key=len, reverse=True):
        s = s.replace(zh_unit, zh_unit_map[zh_unit])

    return s


def main():
    id_results = json.load(open(args.id_results))
    ood_results = json.load(open(args.ood_results)) if args.ood_results is not None else None
    if ood_results is not None:
        results = id_results | ood_results
    else:
        results = id_results
    all_datas = json.load(open('../../data/annotation_question_formula_examples.json'))
    all_datas = [{'id': all_datas[idx]['id'], 'original_data': all_datas[idx]} for idx in range(len(all_datas))]
    all_datas = dict([(s['id'], s) for s in all_datas])

    extra_id_keys = ['8826426_1', '3779814_1', '40725571_3', '51239820_3', '2349432_2', '50124009_2', '8418790_3', '21110353_4', '51051754_2', '51087057_4', '52264763_2',
                     '52680030_2', '51988360_3', '11408574_2', '50076261_2']

    id_keys = set(id_results.keys()) | set(extra_id_keys)
    print(f'{len(id_keys)=}')
    if ood_results is not None:
        ood_keys = set(ood_results.keys()) - set(extra_id_keys)
        print(f'{len(ood_keys)=}')
    else:
        ood_keys = {}

    id_correct_count = 0
    ood_correct_count = 0
    id_results_w_answer = {}
    ood_results_w_answer = {}
    for key in tqdm(results, total=len(results)):
        if key not in all_datas:
            print(key)
            continue
        if isinstance(results[key], dict):
            if 'text' in results[key]['raw_response']:
                results[key] = results[key]['raw_response']['text']
            else:
                results[key] = results[key]['raw_response']['choices'][0]['message']['content']
        pred = extract_answer(results[key])
        answer = process_gold_answer(all_datas[key]['original_data']['answer'])
        numbat_resp = subprocess.run(['numbat'], input=f'abs(({pred}-{answer})/({answer}))<1%', text=True, capture_output=True, encoding='UTF-8', timeout=10)
        is_correct = 'true' in str(numbat_resp.stdout).lower()
        if is_correct:
            if key in id_keys:
                id_correct_count += 1
            else:
                ood_correct_count += 1
        if key in id_keys:
            id_results_w_answer[key] = {'response': results[key], 'pred': pred, 'answer': answer, 'is_correct': is_correct}
        else:
            ood_results_w_answer[key] = {'response': results[key], 'pred': pred, 'answer': answer, 'is_correct': is_correct}

    assert len(id_results_w_answer) == len(id_keys)
    assert len(ood_results_w_answer) == len(ood_keys) or ood_results is None

    print(f'id acc={id_correct_count / len(id_results_w_answer)}')
    if ood_results is not None:
        print(f'ood acc={ood_correct_count / len(ood_results_w_answer)}')

    with open(args.id_results.replace('.json', '.with_answer.json'), 'w', encoding='utf-8') as f:
        json.dump(id_results_w_answer, f, ensure_ascii=False, indent=4)
    if ood_results is not None:
        with open(args.ood_results.replace('.json', '.with_answer.json'), 'w', encoding='utf-8') as f:
            json.dump(ood_results_w_answer, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()
