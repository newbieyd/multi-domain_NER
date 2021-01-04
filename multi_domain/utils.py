import random
import torch
import numpy as np

# 设置随机种子，一旦固定种子，后面依次生成的随机数其实都是固定的
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# 计算评价指标：准确率，召回率，F1值
def calculate(data):
    p = -1
    r = -1
    f1 = -1
    if data[0] > 0:
        p = data[2] / data[0]
    if data[1] > 0:
        r = data[2] / data[1]
    if p != -1 and r != -1 and p + r != 0:
        f1 = 2 * p * r / (p + r)
    return p, r, f1
