import argparse
import concurrent
import json
import math
import os.path
import random
from time import sleep

from tqdm import tqdm
from zhipuai import ZhipuAI

api_key = 'GLM4 api key'

client = ZhipuAI(api_key=api_key)

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, required=True)
parser.add_argument('--output_file', type=str, default=None)
parser.add_argument('--model_name', type=str, choices=['glm-4-flash', 'glm-4-plus'], required=True)
args = parser.parse_args()

if args.output_file is None:
    args.output_file = args.input_file.replace('.json', '.results.json')
    assert args.input_file != args.output_file, f'{args.input_file=}\t{args.output_file=}'


def do_request(prompt, id):
    sleep(int(2 * (random.random() + 2)))
    try:
        response = client.chat.completions.create(
            model=args.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=1024,
        )
    except Exception as e:
        print(e)
        return {'prompt': prompt, 'failed': True}
    return response.choices[0].message.content, id


if __name__ == '__main__':
    max_threads = 10

    output_datas = json.load(open(args.output_file)) if os.path.exists(args.output_file) else {}

    datas = json.load(open(args.input_file))
    datas = list(filter(lambda x: 'id' in x, datas))
    ids = [d['id'] for d in datas if d['id'] not in output_datas or d['id'] in output_datas and 'failed' in output_datas[d['id']]]
    prompts = [d['prompt'] for d in datas if d['id'] not in output_datas or d['id'] in output_datas and 'failed' in output_datas[d['id']]]

    batch_size = 50
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            batch_results = []
            futures = {executor.submit(do_request, **{'prompt': prompt, 'id': id}): prompt for prompt, id in zip(batch, batch_ids)}

            for future in tqdm(concurrent.futures.as_completed(futures), desc=f'batch {int(i / batch_size) + 1}/{math.ceil(len(prompts) / batch_size)}', total=len(batch)):
                prompt = futures[future]
                result, id = future.result()
                batch_results.append(result)
                output_datas[id] = result
            # output_datas = output_datas | dict(zip(batch_ids, batch_results))

            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(output_datas, f, ensure_ascii=False, indent=4)
