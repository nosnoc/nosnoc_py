import numpy as np
import casadi as ca


def validate(obj: object) -> bool:
    for attr, expected_type in obj.__annotations__.items():
        value = getattr(obj, attr)
        if not isinstance(value, expected_type):
            raise TypeError(
                f"object {type(obj)} does not match type annotations. Attribute {attr} should be of type {expected_type}, got {type(value)}"
            )
    return


def print_dict(input: dict):
    out = ''
    for k, v in input.items():
        out += f"{k} : {v}\n"
    print(out)
    return


def casadi_length(x) -> int:
    return x.shape[0]


def print_casadi_vector(x):
    for i in range(casadi_length(x)):
        print(x[i])


def casadi_vertcat_list(x):
    result = []
    for el in x:
        result = ca.vertcat(result, el)
    return result


def casadi_sum_list(input: list):
    result = input[0]
    for v in input[1:]:
        result += v
    return result


# Note this is not generalized, it expects equivalent depth, greater than `layer`
def flatten_layer(L: list, layer=0):
    if layer == 0:
        # Check if already flat
        if any(isinstance(e, list) for e in L):
            return sum(L, [])
        else:
            return L
    else:
        return [flatten_layer(l, layer=layer - 1) for l in L]


# Completely flattens the list
def flatten(L):
    if isinstance(L, list):
        return [a for i in L for a in flatten(i)]
    else:
        return [L]


def increment_indices(L, inc):
    L_new = []
    for e in L:
        if isinstance(e, int):
            L_new.append(e + inc)
        elif isinstance(e, list):
            L_new.append(increment_indices(e, inc))
        else:
            raise ValueError('Not a nested list of integers')
    return L_new

def create_empty_list_matrix(list_dims: tuple):
    return np.empty(list_dims + (0,), dtype=int).tolist()
