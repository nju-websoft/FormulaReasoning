import re
import subprocess


def preprocess_expression(expression):
    expression = re.sub(r'[，、：；。]+', '', expression)
    expression = expression.replace('min(', 'minimum(').replace('max(', 'maximum(')
    expression = expression.replace('{', '(').replace('}', ')').replace('[', '(').replace(']', ')').replace('（', '(').replace('）', ')')
    expression = expression.replace('﹣', '-').replace('−', '-').replace('－', '-').replace('•', '*')
    expression = expression.replace('r', '').replace('sqt(', 'sqrt(')
    expression = expression.replace('％', '%').replace('°C', 'K').replace('℃', 'K').replace('w', 'W').replace('Hz', 's⁻¹').replace('t', 'ton').replace('sqrton(', 'sqrt(')
    expression = expression.replace('度', 'kW·h').replace('元', 'CNY').replace('天', 'day').replace('人', 'person').replace('焦', 'J').replace('千克', 'kg')
    return expression


def postprocess_answer(answer):
    answer = answer.replace('\n', '').replace('_', '')
    answer = answer.replace('K', '℃').replace('ton', 't')
    return answer.strip()


def get_numbat_answer(expression):
    try:
        expression = preprocess_expression(expression)
        response = subprocess.run(['numbat'], input=expression, text=True, capture_output=True, encoding='UTF-8', timeout=10)
        if response.returncode == 0:
            answer = str(response.stdout)
        else:
            answer = None
            # print(response.stderr)
        answer = postprocess_answer(answer)
        return answer
    except Exception as e:
        print(e)
        print(expression)
        return None


def example_to_programs(example):
    formula_list = example['original_data']['formula_list']
    argument_dict = example['original_data']['argument_dict']

    program = []
    program_answer = []
    for formula in formula_list:
        args = re.findall(r'\[(.*?)]', formula)
        args_value = [f"{argument_dict[arg]['数值']} {argument_dict[arg]['单位']}" for arg in args]
        for arg, arg_value in sorted(zip(args, args_value), key=lambda x: len(x[0]), reverse=True):
            formula = formula.replace(arg, arg_value)
        formula = re.sub(r'[\[\]]', '', formula)
        program.append(formula.split('=')[1])
        program_answer.append(formula.split('=')[0])
    return program, program_answer
