import re

from sympy import symbols, Eq, solve, simplify


def transform(formula):
    for s in '+-*/^=':
        formula = formula.replace(s, f' {s} ')

    left, right = formula.split('=')
    left = left.strip()
    right = right.strip()

    right = right.split(' ')
    right = list(filter(lambda x: x not in '()+-*/^', right))

    left_sym, *right_syms = symbols(' '.join([left] + right))
    try:
        equation = Eq(left_sym, simplify(formula.split('=')[1].strip()))
    except ValueError:
        return []

    res = []
    for variable_to_move in right_syms:
        if 'PERCENT' in str(variable_to_move):
            continue
        result = solve(equation, variable_to_move)
        if len(result) == 0:
            continue
        result = str(result[0])

        eq = result.replace('×', ' * ')
        eq = eq.replace('(', ' ( ')
        eq = eq.replace(')', ' ) ')
        for op in '=+-*^/':
            eq = eq.replace(op, f' {op} ')
        while '  ' in eq:
            eq = eq.replace('  ', ' ')  # .replace('( ', '(').replace(' )', ')')

        # f = f'{variable_to_move} = ' + ' '.join(from_infix_to_prefix(list(filter(lambda x: x != '', eq.split(' ')))))
        f = f'{variable_to_move} = ' + ' '.join(list(filter(lambda x: x != '', eq.split(' '))))

        for num in re.findall(r'_(\d+)_PERCENT', f):
            f = f.replace(f'_{num}_PERCENT', f'{num}%')

        res.append(f)
    return res
