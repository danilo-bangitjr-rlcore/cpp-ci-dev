from typing import Callable

import sympy as sy


def _get_tag_names(input_string: str) -> list[str]:
    result: list[str] = []
    in_braces = False
    current_content = ""

    for char in input_string:
        if char == "{":
            in_braces = True
            current_content = ""
        elif char == "}":
            in_braces = False
            result.append(current_content.strip())
        elif in_braces:
            current_content += char

    return result


def _preprocess_expression_string(input_string: str) -> str:
    result = ""
    in_braces = False
    for char in input_string:
        if char == "{":
            in_braces = True
            continue
        elif char == "}":
            in_braces = False
            continue
        elif in_braces:
            char = char.replace("-", "_")

        result += char

    return result


def to_sympy(input_string: str) -> tuple[sy.Expr, Callable, list[str]]:
    """
    The input string might look something like:

    '3*{tag-0} + 2*{tag-1}'

    Note the dash "-" in the tag names could be interpreted as a minus sign.
    This is why we escape them with curly braces "{my-tag}",
    and replace the dash with an underscore "_" when ineracting with sympy.

    Before processing it is called a tag name (e.g. tag-0)
    After processing it is called a symbol name (e.g. tag_0).
    """

    processed_expression = _preprocess_expression_string(input_string)
    expression = sy.sympify(processed_expression)

    # Both are sorted to keep parity between tag names and symbol names
    symbol_names = sorted(expression.free_symbols, key=lambda s: s.name)
    tag_names = sorted(_get_tag_names(input_string))

    lambda_expression: Callable[..., float] = sy.lambdify(symbol_names, expression, "numpy")
    return expression, lambda_expression, tag_names


def is_expression(input_string: str) -> bool:
    """
    Simple check to see if the string should be treated as a _sympy expression_
    """
    if not _is_balanced_braces(input_string):
        raise ValueError("Error: unbalanced curly braces in expression")

    return ("{" in input_string) and ("}" in input_string)

def _is_balanced_braces(input_string: str) -> bool:
    """
    Check if the braces (that we use for escaping the tags) are balanced.
    Nested braces are not allowed
    """
    count = 0
    for c in input_string:
        if c == "{":
            count += 1
        elif c == "}":
            count -= 1
        if count < 0:
            return False

    return count == 0


def _preprocess_tag_names(input_tags: list[str] | str) -> list[str] | str:
    if isinstance(input_tags, list):
        return [tag.replace("-", "_") for tag in input_tags]
    else:
        return input_tags.replace("-", "_")


def is_affine(input_expr: sy.Expr) -> bool:
    try:
        variables = input_expr.free_symbols

        # Conver to polynomial
        poly = sy.Poly(input_expr, *variables)

        # Check if degree is at most 1 for every variable
        for var in variables:
            if poly.degree(var) > 1:
                return False

        # Check if there are mixed terms (like x*y)
        # Note: we might want to permit this in the future
        for monom in poly.monoms():
            if sum(1 for deg in monom if deg > 0) > 1:
                return False

        return True

    # Pyright does not recognize sy.PolynomialError as a valid Exception
    except Exception:
        return False


def eval_expression(expr: sy.Expr, tag_values: dict[str, float]) -> float:
    subs_values = {}
    for key, value in tag_values.items():
        subs_values[_preprocess_tag_names(key)] = value

    result = sy.N(expr.subs(subs_values))
    return float(result)
