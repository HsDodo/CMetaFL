from .utils import *
import torch
import fedml.core.security.common.utils as utils
import logging
def Krum_defense(w_locals,args):  # 注意这里传近来的w_locals是一个元组列表，每个元组包含了 (样本量，模型参数)
    print("########### 开启 Krum_defense 防御 #############")

    num_clients = len(w_locals)
    dist_matrix = np.zeros((num_clients, num_clients)) # 每个客户端之间的距离,用 n*n的矩阵表示
    # 计算权重之间的距离,
    for i in range(num_clients):
        (_ , weights_i) = w_locals[i]   # weights_i 是一个字典，包含了模型的参数
        for j in range(i + 1, num_clients):            # 因为是对称矩阵，只计算一个三角形就行
            (_ , weights_j) = w_locals[j]
            dist = 0
            for k in weights_i.keys():
                if k.endswith("weight"):
                    # 将tensor向量展平
                    flatten_weights_i = weights_i[k].flatten()
                    flatten_weights_j = weights_j[k].flatten()
                    dist = dist + euclidean_distance(flatten_weights_i, flatten_weights_j)
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    min_sum_dist = float('inf')
    selected_index = -1
    for i in range(num_clients):
        sorted_indices = np.argsort(dist_matrix[i])
        sum_dist = np.sum(dist_matrix[i, sorted_indices[0:(num_clients - args.num_attackers)]])
        if sum_dist < min_sum_dist:
            min_sum_dist = sum_dist
            selected_index = i
    (_,weights_best) = w_locals[selected_index]
    return weights_best

# -------------------------mahala -------------------------
def DWAMA(
    w_locals: list[Tuple[float, OrderedDict]],
    w_globals: Any = None,
    round_idx: int = 0,
    args=None):
    w_locals_flatten = {}
    (sample_num, averaged_params) = w_locals[0]
    for k in averaged_params.keys():
        w_locals_flatten[k] = []
        for i in range(0, len(w_locals)):
            local_sample_num, local_params = w_locals[i]
            w_locals_flatten[k].append(local_params[k].numpy())
    mahal_score = get_mahalanobis_score_for_clients_spilt(w_locals,args)  # 获取每个客户端的马氏距离得分
    adjusted_params, anomalies_idx,upper_bound = adjust_params(mahal_score, w_locals, w_globals, round_idx, args)  # 调整参数
    adjusted_agg_weight = adjust_weights(w_locals, round_idx, anomalies_idx,args)  # 调整之后的聚合权重
    return adjusted_params, adjusted_agg_weight , mahal_score  # 返回调整之后的参数 和 聚合权重
# --------------------------------- -------------------------
def Median_defense(w_locals,args):
    print("########### 开启 Median_defense 防御 #############")

    num_clients = len(w_locals)
    weights = {}
    for k in w_locals[0][1].keys():
        # 将参数进行展平并堆叠成二维数组
        local_weights_list = []
        for i in range(num_clients):
            local_weights_list.append(w_locals[i][1][k].flatten())
        if not all(len(w.shape) == 1 for w in local_weights_list):
            raise ValueError("All input arrays must have the same shape except for the stacking axis.")
        # 堆叠所有客户端的展平参数
        stacked_weights = np.stack(local_weights_list, axis=0)
        # 计算中位数
        median_weights = np.median(stacked_weights, axis=0)
        w = torch.tensor(median_weights).view(w_locals[0][1][k].shape)
        # for i in range(num_clients):
        #     # 用中位数替换每个客户端的模型更新
        #     w_locals[i][1][k] = w
        weights[k] = w
    return weights


def Trimmed_mean_defense(w_locals,args):
    """
    使用Trimmed Mean方法聚合模型更新。
    :param w_locals: 一个包含所有客户端模型更新的列表。
    :param trim_ratio: 要剔除的极端值比例，范围在0到0.5之间。
    :return: 聚合后的模型更新。
    """
    print("########### 开启Trimmed_mean_defense 防御 (trim_ratio = {})#############".format(args.trim_ratio))

    if not 0 <= args.trim_ratio <= 0.5:
        raise ValueError("Trim ratio must be between 0 and 0.5")
    num_clients = len(w_locals)
    num_trim = int(num_clients * args.trim_ratio)
    weights = {}
    trimmed_weights = {}
    for k in w_locals[0][1].keys():
        # 将参数进行展平并堆叠成二维数组
        local_weights_list = []
        for i in range(num_clients):
            local_weights_list.append(w_locals[i][1][k].flatten())  # 将参数展平
        if not all(len(w.shape) == 1 for w in local_weights_list):
            raise ValueError("All input arrays must have the same shape except for the stacking axis.")
        # 堆叠所有客户端的展平参数
        stacked_weights = np.stack(local_weights_list, axis=0)
        trimmed_weights[k] = np.sort(stacked_weights, axis=0)[num_trim:-num_trim]
        # 计算剔除极端值后的模型更新的中位数
        w = np.mean(trimmed_weights[k], axis=0)
        # 将 w 转成tensor向量
        w = torch.tensor(w).view(w_locals[0][1][k].shape)
        # for i in range(num_clients):
        #     # 用中位数替换每个客户端的模型更新
        #     w_locals[i][1][k] = w
        weights[k] = w
    return weights

def norm_clip_defense(w_locals,w_globals,args):
    print("########### 开启norm_clip_defense防御 (Bound = {})#############".format(args.norm_bound))
    global_model = w_globals
    vec_global_w = utils.vectorize_weight(global_model)
    new_grad_list = []
    for (sample_num, local_w) in w_locals:
        vec_local_w = utils.vectorize_weight(local_w)
        clipped_weight_diff = get_clipped_norm_diff(vec_local_w, vec_global_w,args)
        clipped_w = get_clipped_weights(local_w, global_model, clipped_weight_diff)
        new_grad_list.append((sample_num, clipped_w))
    # return base_aggregation_func(self.config, new_grad_list)  # avg_params
    return new_grad_list



def Mutil_Krum(w_locals: List[Tuple[float, OrderedDict]],args):
    krum_param_m = 50  #  krum_param_m = 1: krum; krum_param_m > 1: multi-krum
    num_client = len(w_locals)
    # in the Krum paper, it says 2 * byzantine_client_num + 2 < client #
    if not 2 * args.num_attackers + 2 <= num_client - krum_param_m:
        raise ValueError(
            "byzantine_client_num conflicts with requirements in Krum: 2 * byzantine_client_num + 2 < client number - krum_param_m"
        )
    vec_local_w = [
        utils.vectorize_weight(w_locals[i][1])
        for i in range(0, num_client)
    ]
    krum_scores = compute_krum_score(vec_local_w,args)
    score_index = torch.argsort(
        torch.Tensor(krum_scores)
    ).tolist()  # indices; ascending
    score_index = score_index[0 : krum_param_m]
    return [w_locals[i] for i in score_index]



