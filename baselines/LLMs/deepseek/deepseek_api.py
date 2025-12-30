import argparse
import concurrent
import json
import math
import os.path

from tqdm import tqdm
import dashscope
from http import HTTPStatus
from openai import OpenAI

api_key = 'deepseek-api-key'
client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, required=True)
parser.add_argument('--output_file', type=str, default=None)
parser.add_argument('--model_name', type=str, choices=['deepseek-reasoner'], required=True)
args = parser.parse_args()
model_name = args.model_name
if args.output_file is None:
    args.output_file = args.input_file.replace('.json', f'.results_{model_name}.json')
    assert args.input_file != args.output_file, f'{args.input_file=}\t{args.output_file=}'


def do_request(prompt):
    # Please install OpenAI SDK first: `pip3 install openai`
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt},
                ],
            temperature=0,
            stream=False
            )
        return response.choices[0].message.content
    except Exception as e:
        print(e)
        return {'prompt': prompt, 'failed': True}


if __name__ == '__main__':
    max_threads = 1

    output_datas = json.load(open(args.output_file,encoding='UTF-8')) if os.path.exists(args.output_file) else {}

    datas = json.load(open(args.input_file, encoding='utf-8'))
    ids = [d['id'] for d in datas if d['id'] not in output_datas or isinstance(output_datas[d['id']],dict)]
    prompts = [d['prompt'] for d in datas if d['id'] not in output_datas or isinstance(output_datas[d['id']],dict)]

    batch_size = 20
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            batch_results = []
            futures = {executor.submit(do_request, prompt): prompt for prompt in batch}

            for future in tqdm(concurrent.futures.as_completed(futures), desc=f'batch {int(i / batch_size) + 1}/{math.ceil(len(prompts) / batch_size)}', total=len(batch)):
                prompt = futures[future]
                result = future.result()
                if not isinstance(result,dict):
                    batch_results.append(result)

            output_datas = output_datas | dict(zip(batch_ids, batch_results))

            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(output_datas, f, ensure_ascii=False, indent=4)
