
import logging
import math
import numpy as np
from scipy.spatial.distance import mahalanobis
from tqdm import tqdm
import torch
import fedml.core.security.common.utils as _utils



def euclidean_distance(x, y):
    return np.linalg.norm(x - y)


def get_mahalanobis_score_for_clients_spilt(w_locals,args):
    d_mahal_all_layer = {}
    mahal_score = []
    (sample_num, averaged_params) = w_locals[0]
    client_num = len(w_locals)
    d_mahal_all_layer = get_split_mahalanobis_distances(w_locals,args)  # 获取每个分片 每个客户端的马氏距离
    d_mahal_all_layer_norm = {}
    for k in d_mahal_all_layer.keys():
        if d_mahal_all_layer[k] != None:
            d_mahal_all_layer_norm[k] = normalize(d_mahal_all_layer[k],axis=1)
    for i in range(client_num):
        score = 0
        for k in d_mahal_all_layer.keys():
            if k.endswith('weight'):  # 只统计 weight 的马氏距离，bias的变化太大不统计
                if d_mahal_all_layer[k] != None:
                    score += d_mahal_all_layer_norm[k][0][i]
        mahal_score.append(np.abs(score))
    return mahal_score  # 这里返回的是每个客户端的马氏距离得分 是一个 list


def adjust_params(mahal_score, w_locals, w_globals,round_idx, args):
    # 第 i 个客户端如果有问题，是进行调整还是直接剔除？
    adjusted_local_weight = []
    # 计算马哈拉诺比斯距离的四分位数
    Q1 = np.percentile(mahal_score, 25)
    Q3 = np.percentile(mahal_score, 75)
    # 计算四分位距
    IQR = Q3 - Q1
    threshold_init = args.threshold_init
    threshold_min = args.threshold_min
    beta = threshold_init - ((threshold_init-threshold_min)/args.comm_round) * round_idx
    upper_bound = Q3 + beta*IQR
    print(f"上界:{upper_bound}")
    detected_malicious_client = []
    for i in range(len(mahal_score)):
        if mahal_score[i] > upper_bound:
            detected_malicious_client.append(i)
    print(detected_malicious_client)
    anomalies_idx = [0 for i in range(len(w_locals))]
    for i in range(0, len(w_locals)):
        sample_num, params = w_locals[i]
        if (len(mahal_score) != 0):
            if mahal_score[i] != None and abs(mahal_score[i]) > upper_bound:
                params = w_globals
                anomalies_idx[i] = 1
                logging.info("客户端{}参数调整：local_weight --> global_weights ".format(i))
        adjusted_local_weight.append((sample_num, params))
    return adjusted_local_weight, anomalies_idx,upper_bound


def adjust_weights(w_locals, round_idx, anomalies_idx=None,args=None):
    num_locals = len(w_locals)
    client_weights = [1 for i in range(num_locals)]
    decay_rate = math.exp(-args.alpha / (round_idx + 1))
    for i in range(num_locals):
        if anomalies_idx[i] == 1:
            client_weights[i] *= decay_rate
    sum = np.sum(client_weights)
    client_weights = [client_weights[i] / sum * 100 for i in range(num_locals)]
    return client_weights


def get_split_mahalanobis_distances(w_locals,args):  # 获取每个分片的马氏距离
    w_locals_flatten = {}
    d_mahal = {}
    (sample_num, averaged_params) = w_locals[0]
    num_clients = len(w_locals)
    for k in averaged_params.keys():
        w_locals_flatten[k] = []
        for i in range(0, len(w_locals)):
            local_sample_num, local_params = w_locals[i]
            # 将每个客户端的参数tensor展开成numpy数组，放入列表中
            w_locals_flatten[k].append(local_params[k].numpy().ravel())
    # w_locals_flatten 为展开的参数向量,先考虑特征维度大于样本数的情况
    count = 0
    for k in averaged_params.keys():
        if k.endswith("weight") and count < 2:
            count = count + 1
            # w_means = np.mean(w_locals_flatten[k], axis=0)  # 这里要用 中位数代替
            median = np.median(w_locals_flatten[k], axis=0)  # 用 中值来替代 均值来做
            num_features = len(w_locals_flatten[k][0])
            d_mahal[k] = []
            # 这里开始分片计算马氏距离
            num_slices = calculate_num_slices_based_on_cov_matrix_size(num_features,args)  # 计算分片个数
            slice_size = num_features // num_slices  # 每个分片的大小
            all_slice_mahal = []
            for i in tqdm(range(num_slices), ncols=80, desc=f"{k}层分片:"):  # 将 w_locals 分片进行处理
                # 计算分片的协方差矩阵
                epsilon = 1e-5
                start = i * slice_size
                end = min((i + 1) * slice_size, num_features)
                w_locals_flatten_k_array = np.array(w_locals_flatten[k])
                w_cov = np.cov(w_locals_flatten_k_array[:, start:end], rowvar=False)
                lambda_identity = np.eye(end - start) * epsilon
                w_cov = w_cov + lambda_identity
                w_cov_inv = np.linalg.inv(w_cov)
                std = np.nanstd(w_locals_flatten_k_array[:, start:end], axis=0)
                std[std == 0] = 1
                w_locals_normalized = (w_locals_flatten_k_array[:, start:end] - np.nanmean(w_locals_flatten_k_array[:, start:end], axis=0)) / std
                slice_i_mahal_all_clients = []
                for j in range(0, len(w_locals)):
                    client_mahal = mahalanobis(w_locals_normalized[j], median[start:end], w_cov_inv)
                    slice_i_mahal_all_clients.append(client_mahal)
                slice_i_mahal_all_clients_normalized = normalize(np.array(slice_i_mahal_all_clients))
                all_slice_mahal.append(slice_i_mahal_all_clients_normalized)
            w_k_mahal = slice_mahal_aggrate(all_slice_mahal)
            d_mahal[k].append(w_k_mahal)
    return d_mahal

def slice_mahal_aggrate(slice_mahal):
    aggrated_mahal = []
    for i in range(len(slice_mahal)):
        for j in range(len(slice_mahal[i])):
            if i == 0:
                aggrated_mahal.append(np.square(slice_mahal[i][j]))
            else:
                aggrated_mahal[j] += np.square(slice_mahal[i][j])
    return aggrated_mahal

def normalize(data, axis=0):
    mean = np.mean(data, axis=axis)
    std = np.std(data, axis=axis) # 按列求标准差
    std = np.where(std == 0, 1, std)
    return (data - mean) / std

def calculate_num_slices_based_on_cov_matrix_size(total_features,args):
    """
    根据最大协方差矩阵大小计算分片个数。

    # 假设有 1GB 可用内存
    :param total_features: 数据集中的总特征数。
    :param max_cov_matrix_size: 可处理的最大协方差矩阵的大小（即最大的特征数）。
    :return: 计算得到的分片个数。
    """
    # 确保每个分片的特征数不超过协方差矩阵的最大允许大小
    available_memory = args.available_memory * 1024 * 1024 * 1024  # 1GB in bytes
    element_memory = 4  # 4 bytes per float32
    max_features_per_slice = int(np.sqrt(available_memory / element_memory))
    # max_features_per_slice = int(np.sqrt(max_matrix_size))
    # 计算所需的分片个数
    if total_features <= max_features_per_slice:
        num_slices = 1
    else:
        num_slices = (total_features + max_features_per_slice - 1) // max_features_per_slice
    return num_slices


# ============== norm clip ==================
def get_clipped_norm_diff(vec_local_w, vec_global_w,args):
    vec_diff = vec_local_w - vec_global_w
    weight_diff_norm = torch.norm(vec_diff).item()
    clipped_weight_diff = vec_diff / max(1, weight_diff_norm / args.norm_bound)
    return clipped_weight_diff

def get_clipped_weights(local_w, global_w, weight_diff): #
    #  rule: global_w + clipped(local_w - global_w)
    recons_local_w = {}
    index_bias = 0
    for item_index, (k, v) in enumerate(local_w.items()):
        if _utils.is_weight_param(k):
            recons_local_w[k] = (
                weight_diff[index_bias : index_bias + v.numel()].view(v.size())
                + global_w[k]
            )
            index_bias += v.numel()
        else:
            recons_local_w[k] = v
    return recons_local_w


def compute_krum_score(vec_grad_list,args):
    krum_scores = []
    num_client = len(vec_grad_list)
    for i in range(0, num_client):
        dists = []
        for j in range(0, num_client):
            if i != j:
                dists.append(
                    _utils.compute_euclidean_distance(
                        vec_grad_list[i], vec_grad_list[j]
                    ).item() ** 2
                )
        dists.sort()  # ascending
        score = dists[0 : num_client - args.num_attackers - 2]
        krum_scores.append(sum(score))
    return krum_scores

def is_weight_param(k):
    return (
            "running_mean" not in k
            and "running_var" not in k
            and "num_batches_tracked" not in k
    )

