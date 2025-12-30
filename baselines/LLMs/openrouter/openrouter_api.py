import argparse
import concurrent
import json
import math
import os.path
import random
import time
from time import sleep
import requests
from tqdm import tqdm
from openai import OpenAI

key = 'openrouter-api-key'

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, required=True)
parser.add_argument('--output_file', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--name', type=str, required=True)
args = parser.parse_args()
name = args.name
model_name = name.split('/')[-1].split(':free')[0]
if args.output_file is None:
    args.output_file = args.input_file.replace('.json', f'.results_{model_name}.json')
    assert args.input_file != args.output_file, f'{args.input_file=}\t{args.output_file=}'

args.raw_output_file = args.input_file.replace('.json', f'.raw_results_{model_name}.json')

def get_answer(prompt,id):
    # while True:
        try:
            time.sleep(3)
            headers = {
                'Authorization': f'Bearer {key}',
                'Content-Type': 'application/json',
            }
            response = requests.post('https://openrouter.ai/api/v1/chat/completions',headers=headers, json={
                'model': name,
                'seed': 0,
                'temperature':0,
                'messages': [
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                'provider': {
                    'order': [
                        'Azure'
                    ]
                }
            })
            results = response.json()
            return results,results['choices'][0]['message']['content'], id
        except:
            pass
            return {'prompt': prompt, 'failed': True}


if __name__ == '__main__':
    max_threads = 2

    output_datas = json.load(open(args.output_file,encoding='UTF-8')) if os.path.exists(args.output_file) else {}
    raw_output_datas = json.load(open(args.raw_output_file,encoding='UTF-8')) if os.path.exists(args.raw_output_file) else {}

    datas = json.load(open(args.input_file,encoding='UTF-8'))
    datas = list(filter(lambda x: 'id' in x, datas))
    ids = [d['id'] for d in datas if d['id'] not in output_datas]
    prompts = [d['prompt'] for d in datas if d['id'] not in output_datas]

    batch_size = args.batch_size

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            batch_results = []
            futures = {executor.submit(get_answer, **{'prompt': prompt, 'id': id}): prompt for prompt, id in zip(batch, batch_ids)}

            for future in tqdm(concurrent.futures.as_completed(futures), desc=f'batch {int(i / batch_size) + 1}/{math.ceil(len(prompts) / batch_size)}', total=len(batch)):
                prompt = futures[future]
                r = future.result()
                if isinstance(r, dict):
                    continue
                result, msg, id = r
                res = {'usage':result['usage'],'res':str(result)}
                batch_results.append(msg)
                output_datas[id] = msg
                raw_output_datas[id] = res
            # output_datas = output_datas | dict(zip(batch_ids, batch_results))

            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(output_datas, f, ensure_ascii=False, indent=4)
            with open(args.raw_output_file, 'w', encoding='utf-8') as f:
                json.dump(raw_output_datas, f, ensure_ascii=False, indent=4)
