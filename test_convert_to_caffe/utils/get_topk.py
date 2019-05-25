import numpy as np


def get_topk(prob, k=5):
    size = prob.shape[0]
    k = min(k, size)
    sorted_idx = np.argsort(prob)[::-1]
    topk_idx = sorted_idx[:k]
    topk_prob = prob[sorted_idx[:k]]
    return topk_idx, topk_prob
