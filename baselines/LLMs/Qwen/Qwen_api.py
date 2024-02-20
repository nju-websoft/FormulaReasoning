import argparse
import concurrent
import json
import math
import os.path

from tqdm import tqdm
import dashscope
from http import HTTPStatus

api_key = 'aliyun api key here'
dashscope.api_key = api_key

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, required=True)
parser.add_argument('--output_file', type=str, default=None)
args = parser.parse_args()

if args.output_file is None:
    args.output_file = args.input_file.replace('.json', '.results_qwen_max.json')
    assert args.input_file != args.output_file, f'{args.input_file=}\t{args.output_file=}'


def do_request(prompt):
    try:
        response = dashscope.Generation.call(
            model=dashscope.Generation.Models.qwen_max,
            prompt=prompt
        )
        # The response status_code is HTTPStatus.OK indicate success,
        # otherwise indicate request is failed, you can get error code
        # and message from code and message.
        if response.status_code == HTTPStatus.OK:
            return response.output['text']
        else:
            return {'prompt': prompt, 'failed': True}
    except Exception as e:
        print(e)
        return {'prompt': prompt, 'failed': True}


if __name__ == '__main__':
    max_threads = 1

    output_datas = json.load(open(args.output_file)) if os.path.exists(args.output_file) else {}

    datas = json.load(open(args.input_file, encoding='utf-8'))
    ids = [d['id'] for d in datas if d['id'] not in output_datas]
    prompts = [d['prompt'] for d in datas if d['id'] not in output_datas]

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
                batch_results.append(result)

            output_datas = output_datas | dict(zip(batch_ids, batch_results))

            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(output_datas, f, ensure_ascii=False, indent=4)
