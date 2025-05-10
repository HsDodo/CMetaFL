
import logging
from .utils import *

# 高斯分布攻击，使用高斯分布（正态分布）来生成异常模型更新，用生成的高斯噪声来替换原始的模型更新
# 这样，恶意节点的更新将模拟正常节点的更新，但实际上是随机噪声
def gaussian_attack(w_locals,args):
    logging.info("########### 开启 gaussian攻击 (rate = {}) #############".format(args.attack_rate))
    client_num_per_round = args.client_num_per_round
    attack_ratio = args.attack_ratio
    attack_num = (int)(client_num_per_round * attack_ratio)
    if args.random_attack: # 随机选取客户端进行攻击
        client_attack_idx = random.sample(range(client_num_per_round),min(attack_num,client_num_per_round))
    else:
        if hasattr(args, "attack_client_list"):
            client_attack_idx = args.attack_client_list
        else:
            client_attack_idx = random.sample(range(client_num_per_round),min(attack_num, client_num_per_round))
    client_idx = []
    (_,averaged_params) = w_locals[client_attack_idx[0]]
    attack_w = copy.deepcopy(w_locals)
    for idx in client_attack_idx:
        (_, local_params) = w_locals[idx]
        (_, attack_params) = attack_w[idx]
        for k in averaged_params.keys():
            flatten_params = torch.cat([vv.reshape((-1, 1)) for vv in local_params[k]], dim=0)
            std = torch.std(flatten_params, unbiased=False).item()
            for j in range(0, len(local_params[k])):
                attack_params[k][j] = torch.tensor(np.random.normal(0, std, size=local_params[k][j].shape).astype('float32')).to(local_params[k][j].device)
    logging.info("被攻击的客户端索引 = %s" % str(client_attack_idx))
    return attack_w,client_attack_idx
#
def scaling_attack(w_locals,args):
    logging.info("########### 开启 scaling_attack 攻击  #############".format(args.attack_rate))
    client_num_per_round = args.client_num_per_round
    attack_rate = args.attack_rate
    attack_num = (int)(client_num_per_round * attack_rate)
    if hasattr(args,"attack_client_list"):
        client_attack_idx = args.attack_client_list
    else:
        client_attack_idx = random.sample(range(client_num_per_round),min(attack_num,client_num_per_round)) # 被攻击的客户端索引
    (_,averaged_params) = w_locals[client_attack_idx[0]]
    attack_w = copy.deepcopy(w_locals)
    for k in averaged_params.keys():
        if k.endswith("weight"):
            for idx in range(args.client_num_per_round):
                if idx in client_attack_idx:
                    (_,local_params) = w_locals[idx]
                    (_, attack_params) = attack_w[idx]
                    #  随机攻击某些模型参数
                    for j in range(0, len(attack_params[k])):
                        attack_params[k][j] = local_params[k][j] * args.scaling_factor
    logging.info("被攻击的客户端索引(0 开始的) = %s" % str(client_attack_idx))
    return attack_w,client_attack_idx

def reverse_attack(w_locals,args):
    logging.info("########### 开启 reverse_attack 攻击 #############".format(args.attack_rate))
    client_num_per_round = args.client_num_per_round
    attack_rate = args.attack_rate
    attack_num = (int)(client_num_per_round * attack_rate)
    if hasattr(args,"attack_client_list"):
        client_attack_idx = args.attack_client_list
    else:
        client_attack_idx = random.sample(range(client_num_per_round),min(attack_num,client_num_per_round)) # 被攻击的客户端索引
    (_,averaged_params) = w_locals[client_attack_idx[0]]
    attack_w = copy.deepcopy(w_locals)
    for idx in client_attack_idx:
        (_, local_params) = w_locals[idx]
        (_, attack_params) = attack_w[idx]
        for k in averaged_params.keys():
            if k.endswith("weight"):
                attack_params[k] = -local_params[k]

    logging.info("被攻击的客户端索引 = %s" % str(client_attack_idx))
    return attack_w,client_attack_idx

# reverse_attack 和 scaling_attack 的混合攻击
def mixed_attack(w_locals,args):
    attack_prob = 0.8
    logging.info("########### 开启 mixed_attack 攻击 #############".format(args.attack_rate))
    client_num_per_round = args.client_num_per_round
    attack_rate = args.attack_rate
    attack_num = (int)(client_num_per_round * attack_rate)
    if hasattr(args,"attack_client_list"):
        client_attack_idx = args.attack_client_list
    else:
        client_attack_idx = random.sample(range(client_num_per_round),min(attack_num,client_num_per_round)) # 被攻击的客户端索引
    (_,averaged_params) = w_locals[client_attack_idx[0]]
    attack_w = copy.deepcopy(w_locals)
    for k in averaged_params.keys():
        for idx in range(args.client_num_per_round):
            if idx in client_attack_idx:
                (_,local_params) = w_locals[idx]
                (_, attack_params) = attack_w[idx]
                # 随机攻击某些特征参数
                for j in range(0, len(local_params[k])):
                    random_prob = random.random()
                    if random_prob < attack_prob:
                        attack_params[k][j] = local_params[k][j] * args.scaling_factor
                        attack_params[k][j] = local_params[k][j] * -1
    logging.info("被攻击的客户端索引(0 开始的) = %s" % str(client_attack_idx))
    return attack_w,client_attack_idx
