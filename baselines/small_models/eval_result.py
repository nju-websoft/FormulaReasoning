import argparse
import json
from tqdm import tqdm

from annotation.process_table_qwen_response import get_numbat_answer


def get_formula_list(formula_text):
    text_splits = formula_text.replace(' ', '').split(',')
    return text_splits


def get_argument_dict(argument_map):
    argument_dict = {}
    for key, value in argument_map.items():
        measures = value.replace('，', '').replace(',', '').split('单位:')
        if len(measures) == 2:
            argument_dict[key] = {
                '数值': measures[0],
                '单位': measures[-1]
            }
    return argument_dict


def verify_answer(formula_list, argument_dict, web_answer, target_argument):
    expression_dict = {}
    numbat_answer = 'null'
    for idx, formula in enumerate(formula_list):
        formula_splits = formula.split('=')
        unknown = formula_splits[0]
        expression = formula_splits[-1]
        for arg, measure in argument_dict.items():
            if arg in expression:
                arg_number = measure['数值']
                arg_unit = measure['单位']
                variable = f'(({arg_number}) {arg_unit})'
                expression = expression.replace(arg, variable)
        answer = get_numbat_answer(expression)
        expression_dict[expression] = answer
        if answer == 'null':
            continue
        else:
            numbat_answer = answer

        answer_number, answer_unit = answer.split(' ')
        # 加入中间结果
        if unknown not in argument_dict:
            argument_dict[unknown] = {
                '数值': answer_number,
                '单位': answer_unit,
            }
        else:
            unknown_number = argument_dict[unknown]['数值']
            unknown_unit = argument_dict[unknown]['单位']
            if get_numbat_answer(f'{unknown_number} {unknown_unit} == {answer}') != 'true':
                argument_dict[unknown] = {
                    '数值': answer_number,
                    '单位': answer_unit,
                }

    if numbat_answer != 'null' and get_numbat_answer(f'abs({numbat_answer}-{web_answer})/({web_answer})<1%') == 'true':
        pred_flag = True
    else:
        pred_flag = False
    pred_argument = formula_list[-1].split('=')[0]
    return {
        'argument_flag': target_argument in pred_argument or pred_argument in target_argument,
        'target_argument': target_argument,
        'pred_argument': pred_argument,
        'formula_list': formula_list,
        'argument_dict': argument_dict,
        'expression_dict': expression_dict,
        'answer_flag': pred_flag,
        'answer': web_answer,
        'pred_answer': numbat_answer,
    }


parser = argparse.ArgumentParser()
parser.add_argument('--id_results', type=str, required=True)
parser.add_argument('--ood_results', type=str, required=True)
args = parser.parse_args()

if __name__ == '__main__':
    id_results = json.load(open(args.id_results, 'r', encoding='utf-8'))
    ood_results = json.load(open(args.ood_results, 'r', encoding='utf-8'))
    results = id_results + ood_results


    id_keys = set([data['id'] for data in id_results])
    print(f'{len(id_keys)=}')

    ood_keys = set([data['id'] for data in ood_results])
    print(f'{len(ood_keys)=}')

    id_argument_correct_count = 0
    id_answer_correct_count = 0
    ood_argument_correct_count = 0
    ood_answer_correct_count = 0

    id_first_try_correct_count = 0
    id_5_tries_correct_count = 0
    ood_first_try_correct_count = 0
    ood_5_tries_correct_count = 0

    for data in tqdm(results):
        data_id = data['id']
        data_answer = data['original_data']['answer']
        preds = data['preds']
        target_argument = data['original_data']['formula_list'][-1].replace('[', '').replace(']', '').split('=')[0]

        is_first_not_none = True
        _5_try_correct = False
        for pred in preds:
            if pred is None:
                continue
            formula_list = get_formula_list(pred['formula'])
            argument_dict = get_argument_dict(pred['symbol_map'])
            eval_result = verify_answer(formula_list, argument_dict, data_answer, target_argument)
            pred = pred | eval_result

            if eval_result['argument_flag']:
                if data_id in id_keys:
                    id_argument_correct_count += 1
                else:
                    ood_argument_correct_count += 1
            if eval_result['answer_flag']:
                _5_try_correct = True
                if data_id in id_keys:
                    id_answer_correct_count += 1
                else:
                    ood_answer_correct_count += 1

            if is_first_not_none:
                if eval_result['answer_flag']:
                    if data_id in id_keys:
                        id_first_try_correct_count += 1
                    else:
                        ood_first_try_correct_count += 1
                is_first_not_none = False

        data['preds'] = preds

        if _5_try_correct:
            if data_id in id_keys:
                id_5_tries_correct_count += 1
            else:
                ood_5_tries_correct_count += 1

    # with open(args.eval_file.replace('.json', '.results.json'), 'w', encoding='UTF-8') as f:
    #     json.dump(pred_results, f, sort_keys=False, indent=4, ensure_ascii=False)

    print(f'{len(id_results)=}')
    print(f'{len(ood_results)=}')

    print(f'arg acc: {id_argument_correct_count / 5 / len(id_results)}')
    print(f'exe acc: {id_answer_correct_count / 5 / len(id_results)}')
    print(f'first try exe acc: {id_first_try_correct_count / len(id_results)}')
    print(f'5 try exe acc: {id_5_tries_correct_count / len(id_results)}')

    print(f'arg acc: {ood_argument_correct_count / 5 / len(ood_results)}')
    print(f'exe acc: {ood_answer_correct_count / 5 / len(ood_results)}')
    print(f'first try exe acc: {ood_first_try_correct_count / len(ood_results)}')
    print(f'5 try exe acc: {ood_5_tries_correct_count / len(ood_results)}')
