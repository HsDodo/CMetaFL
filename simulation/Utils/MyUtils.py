import numpy as np


def str_to_list(str):
    str = str[1:-1]
    str_list = str.split(',')
    return [int(i) for i in str_list]

