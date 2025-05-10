import copy
import random

import numpy as np
import torch
from tqdm import tqdm

from .ModelTrainer import create_model_trainer
from .client import Client
import wandb
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.metrics import silhouette_score


# 聚类联邦学习 Clustered Federated Learning
class CMetaFL(object):
    def __init__(self, args, device, dataset, model):
        self.args = args
        self.device = device
        [
            train_data_num,  # 训练集数量
            test_data_num,  # 测试集数量
            train_data_global,  # 全局训练集
            test_data_global,  # 全局测试集
            train_data_local_num_dict,  # 本地训练集数量字典
            train_data_local_dict,  # 本地训练集字典
            test_data_local_dict,  # 本地测试集字典
            class_num,  # 类别数量
        ] = dataset

        self.train_global = train_data_global
        self.test_global = test_data_global
        self.val_global = None
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num
        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.model_trainer = create_model_trainer(model, args)
        self.model = model
        print("模型(model) = {}".format(model))
        print("设备环境(device) = {}".format(device))
        self._setup_clients(
            self.args, model,train_data_local_num_dict, train_data_local_dict, test_data_local_dict,
        )
        # 存储每个簇的模型
        self.cluster_models = {}
        self.client_cluster_idx_map = []

    # 设置客户端,初始化客户端list
    def _setup_clients(
            self,args,model, train_data_local_num_dict, train_data_local_dict, test_data_local_dict,
    ):
        print("############# 初始化客户端(Start) #############")
        for client_idx in range(self.args.client_num_per_round):
            c = Client(
                client_idx,
                train_data_local_dict[client_idx],
                test_data_local_dict[client_idx],
                train_data_local_num_dict[client_idx],
                self.args,
                self.device,
                create_model_trainer(copy.deepcopy(model), args),
            )
            self.client_list.append(c)

    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        return client_indexes

    # 客户端训练
    def train(self):
        print("全局训练器 = {}".format(self.model_trainer))
        w_global = self.model_trainer.get_model_params()
        for round_idx in tqdm(range(self.args.comm_round), ncols=80, desc="通信轮数", leave=True):
            print("############# 第{}轮通信(Start) #############".format(round_idx))
            w_locals = []
            client_indexes = self._client_sampling(
                round_idx, self.args.client_num_in_total, self.args.client_num_per_round
            )
            print("客户端id = %s" % str(client_indexes))
            for idx, client in enumerate(self.client_list):
                client_idx = client_indexes[idx]
                client.update_local_dataset(
                    client_idx,
                    self.train_data_local_dict[client_idx],
                    self.test_data_local_dict[client_idx],
                    self.train_data_local_num_dict[client_idx]
                )
                # 获取客户端的簇 ID号
                # 若 client_cluster_id 为空，则需要将其置为 0
                if len(self.client_cluster_idx_map) != 0:
                    client_cluster_id = self.client_cluster_idx_map[client_idx]
                    client.cluster_idx = client_cluster_id
                    w_global = self.cluster_models[client_cluster_id]
                # 获取簇模型
                w = client.train_reptile(copy.deepcopy(w_global))
                w_locals.append((client.get_sample_number(), w))
            # 本地客户端训练完后进行聚类，并将各个簇的模型聚合成一个全局模型
            w_global = self._aggregate_with_cluster(w_locals,self.args.cluster_num, self.args.cluster_num,round_idx)
            self.model_trainer.set_model_params(w_global)
            self._compare_global_and_cluster_models(round_idx)
    # 普通加权聚合函数，不含聚类
    def _aggregate(self, w_locals):
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, averaged_params) = w_locals[idx]
            training_num += sample_num
        (sample_num, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            for i in range(len(w_locals)):
                local_sample_num, local_params = w_locals[i]
                weight = (local_sample_num / training_num)
                if i == 0:
                    averaged_params[k] = local_params[k] * weight
                else:
                    averaged_params[k] += local_params[k] * weight
        return averaged_params

    # 平均聚合函数，没有加权
    def _aggregate_noniid_avg(self, w_locals):
        (_, averaged_params) = w_locals[0]
        for k in averaged_params.keys:
            for idx in range(0, len(w_locals)):
                (_, local_params) = w_locals[idx]
                averaged_params[k] += local_params[k]
            averaged_params[k] /= len(w_locals)
        return averaged_params

    ## -------------------- 聚类 ---------------------------------
    #  计算每个客户端模型参数的余弦相似度矩阵
    def _calculate_cosine_similarity(self, w_locals):
        # 提取每个客户端的模型参数的向量表示
        model_params = []
        for _, averaged_params in w_locals:
            # 将 averaged_params 中的所有模型参数展平成一个向量
            param_vector = np.concatenate([param.detach().cpu().flatten().numpy() for param in averaged_params.values()])
            model_params.append(param_vector)

        # 计算余弦相似度矩阵
        cosine_sim_matrix = cosine_similarity(model_params)
        return cosine_sim_matrix

    # 将余弦相似度转化为距离矩阵
    def _calculate_distance_matrix(self, cosine_sim_matrix):
        # 距离矩阵 D = 1 - 相似度矩阵 S
        distance_matrix = 1 - cosine_sim_matrix
        np.fill_diagonal(distance_matrix, 0)  # 确保对角线为零
        return distance_matrix

    # 进行聚类并返回聚完类之后的簇编号，根据客户端的模型参数进行聚类
    def _cluster_clients(self, w_locals, num_clusters, round_idx):
        # 计算每个客户端之间的余弦相似度
        cosine_sim_matrix = self._calculate_cosine_similarity(w_locals)
        # 将相似度转化为距离
        distance_matrix = self._calculate_distance_matrix(cosine_sim_matrix)
        # 使用 KMeans 聚类
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        clusters = kmeans.fit_predict(distance_matrix)  # 获取每个客户端的簇编号
        self.print_clusters(clusters,round_idx)
        print("聚类簇编号Clusters",clusters)
        return clusters

    # def  _cluster_clients(self, w_locals, round_idx):
    #     # 计算每个客户端之间的余弦相似度
    #     cosine_sim_matrix = self._calculate_cosine_similarity(w_locals)
    #     # 将相似度转化为距离
    #     distance_matrix = self._calculate_distance_matrix(cosine_sim_matrix)
    #
    #     # 使用 DBSCAN 聚类
    #     # eps 是距离阈值，min_samples 是形成簇的最小点数
    #     eps = 0.5
    #     min_samples = 2
    #     dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    #     clusters = dbscan.fit_predict(distance_matrix)
    #
    #     # 检查聚类结果是否有效
    #     unique_clusters = set(clusters)
    #     if len(unique_clusters) <= 1:  # 如果只有一个簇或噪声点
    #         print("DBSCAN 无法发现多个簇，请调整参数 eps 或 min_samples。")
    #         return [0] * len(w_locals)  # 将所有客户端分为一个簇
    #
    #     print(f"DBSCAN 发现了 {len(unique_clusters)} 个簇。")
    #     self.print_clusters(clusters, round_idx)
    #
    #     # 如果需要将聚类结果记录到 Wandb 或日志文件
    #     if self.args.enable_wandb:
    #         wandb.log({"num_clusters": len(unique_clusters), "round": round_idx})
    #
    #     return clusters


    # 打印聚类结果
    def print_clusters(self, clusters, round_idx):
        # 将每个簇的客户端索引按簇分组
        cluster_dict = {}
        for client_idx, cluster_id in enumerate(clusters):
            if cluster_id not in cluster_dict:
                cluster_dict[cluster_id] = []
            cluster_dict[cluster_id].append(client_idx)

        # 打印每个簇的客户端
        for cluster_id, client_indices in cluster_dict.items():
            print(f"簇 {cluster_id}: 客户端 {client_indices}")

        # 如果需要将聚类结果记录到 Wandb 或日志文件
        if self.args.enable_wandb:
            for cluster_id, client_indices in cluster_dict.items():
                wandb.log({f"Cluster_{cluster_id}_Clients": str(client_indices), "round": round_idx})

    # 计算聚类轮廓系数
    def _evaluate_clustering(self, w_locals, clusters):
        """
        评估聚类结果的效果
        """
        # 聚类后的轮廓系数评估
        cosine_sim_matrix = self._calculate_cosine_similarity(w_locals)
        score = silhouette_score(cosine_sim_matrix, clusters)
        wandb.log({"silhouette_score": score})
        print(f"聚类轮廓系数: {score}")

        # 对每个簇的客户端进行加权聚合，并记录每个簇的模型
    def _aggregate_with_cluster(self, w_locals, cluster_num, round_idx):
        """
        1. 聚类客户端，将其划分为多个簇。
        2. 每个簇内使用 Reptile 的外循环更新公式，对元模型进行插值更新。
        3. 保存每个簇的聚合模型，并使用 FedAvg 加权聚合所有簇的模型，得到全局模型。
        """
        # 1. 聚类操作（每 10 轮重新聚类一次）
        clusters = []
        if round_idx !=0 and round_idx % 3 == 0:
            clusters = self._cluster_clients(w_locals, cluster_num,round_idx)
            self.client_cluster_idx_map = clusters
        else:
            clusters = self.client_cluster_idx_map

        # 2. 每个簇内进行 Reptile 外循环更新
        cluster_models = {}
        cluster_sample_counts = {}  # 记录每个簇的总样本数

        # 如果聚类为 0 那么全局聚合
        if len(clusters) == 0:
            return self._aggregate(w_locals)

        for cluster_id in set(clusters):
            cluster_params = []  # 簇内客户端参数列表
            cluster_sample_sizes = []

            # 获取当前簇内客户端的参数和样本数
            for idx, cluster in enumerate(clusters):
                if cluster == cluster_id:
                    sample_num, local_params = w_locals[idx]
                    cluster_params.append(local_params)
                    cluster_sample_sizes.append(sample_num)

            # 记录当前簇的样本总量
            total_samples = sum(cluster_sample_sizes)
            cluster_sample_counts[cluster_id] = total_samples

            # Reptile 更新：簇内插值
            base_model = self.cluster_models.get(cluster_id, None)  # 初始簇模型
            # 若簇模型不存在，则将簇内模型直接聚合
            if base_model is None:
                base_model = cluster_params[0]
                for param_name in base_model.keys():
                    # base_model[param_name] *= 0.0  # 初始化为0
                    base_model[param_name] = base_model[param_name].float()  # 转换为浮点型
                    base_model[param_name] *= 0.0  # 初始化为 0
                for param_name in base_model.keys():
                    for i, local_params in enumerate(cluster_params):
                        weight = cluster_sample_sizes[i] / total_samples
                        base_model[param_name] +=  weight * local_params[param_name]
            else:
                meta_lr = self.args.meta_learning_rate  # 外循环学习率
                for param_name in base_model.keys():
                    for i, local_params in enumerate(cluster_params):
                        weight = cluster_sample_sizes[i] / total_samples
                        base_model[param_name] += meta_lr * weight * (local_params[param_name] - base_model[param_name])

            # 保存当前簇的聚合模型
            cluster_models[cluster_id] = base_model
            self.cluster_models[cluster_id] = base_model

        # 3. FedAvg 全局聚合：根据客户端样本总量进行加权聚合
        return self._aggregate(w_locals)

    # 生成验证集
    def _generate_validation_set(self, num_samples=10000):
        test_data_num = len(self.test_global.dataset)
        sample_indices = random.sample(range(test_data_num), min(test_data_num, num_samples))  # 从测试集中随机抽取10000个样本
        subset = torch.utils.data.Subset(self.test_global.dataset, sample_indices)
        sample_testset = torch.utils.data.DataLoader(subset, batch_size=self.args.batch_size)
        self.val_global = sample_testset

    def _local_test_on_all_clients(self, round_idx):
        # print("############# 在所有客户端上进行评估 (第 {} 轮) #############".format(round_idx))
        train_metrics = {"num_samples": [], "losses": [], "num_correct": []}
        test_metrics = {"num_samples": [], "losses": [], "num_correct": []}
        w_global = self.model_trainer.get_model_params()
        client = Client(
            -1,
            self.train_data_local_dict[0],
            self.test_data_local_dict[0],
            self.train_data_local_num_dict[0],
            self.args,
            self.device,
            create_model_trainer(self.model, self.args),
        )
        # for client_idx in tqdm(range(self.args.client_num_in_total), ncols=80, desc="在所有客户端上验证", leave=True):
        for client_idx in range(self.args.client_num_in_total):
            if self.test_data_local_dict[client_idx] is None:
                continue
            client.update_local_dataset(
                client_idx,
                self.train_data_local_dict[client_idx],
                self.test_data_local_dict[client_idx],
                self.train_data_local_num_dict[client_idx],
            )
            client.model_trainer.set_model_params(w_global)
            # 训练
            train_local_metrics = client.local_test(False)  # 训练集上评估,
            train_metrics["num_samples"].append(copy.deepcopy(train_local_metrics["test_total"]))
            train_metrics["num_correct"].append(copy.deepcopy(train_local_metrics["test_correct"]))
            train_metrics["losses"].append(copy.deepcopy(train_local_metrics["test_loss"]))

            # 测试
            test_local_metrics = client.local_test(True)
            test_metrics["num_samples"].append(copy.deepcopy(test_local_metrics["test_total"]))
            test_metrics["num_correct"].append(copy.deepcopy(test_local_metrics["test_correct"]))
            test_metrics["losses"].append(copy.deepcopy(test_local_metrics["test_loss"]))

        # test on training dataset
        train_acc = sum(train_metrics["num_correct"]) / sum(train_metrics["num_samples"])
        train_loss = sum(train_metrics["losses"]) / sum(train_metrics["num_samples"])

        # test on testing dataset
        test_acc = sum(test_metrics["num_correct"]) / sum(test_metrics["num_samples"])
        test_loss = sum(test_metrics["losses"]) / sum(test_metrics["num_samples"])

        stats = {"train_acc": train_acc, "train_loss": train_loss}
        if self.args.enable_wandb:
            wandb.log({"Train/Acc": train_acc, "round": round_idx})
            wandb.log({"Train/Loss": train_loss, "round": round_idx})
        print(stats)

        stats = {"test_acc": test_acc, "test_loss": test_loss}
        if self.args.enable_wandb:
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, "round": round_idx})
        print(stats)

    def _global_test_on_global_model(self, round_idx, w_global):
        client = Client(
            0,
            self.train_data_local_dict[0],
            self.test_data_local_dict[0],
            self.train_data_local_num_dict[0],
            self.args,
            self.device,
            create_model_trainer(self.model, self.args),
        )
        client.update_local_dataset(0, None, self.test_global, None)  # 设置全局测试集
        client.model_trainer.set_model_params(w_global)  # 设置全局模型
        test_metrics = client.local_test(True)  # 在全局测试集上进行评估
        test_acc = test_metrics["test_correct"] / test_metrics["test_total"]
        test_loss = test_metrics["test_loss"] / test_metrics["test_total"]
        if self.args.enable_wandb:
            wandb.log(
                {"Test/Acc (global)": test_acc, "round": round_idx})
            wandb.log(
                {"Test/Loss (global)": test_loss, "round": round_idx})
        print("全局模型在全局测试集上的准确率: {:.4f}, 损失: {:.4f}\n".format(test_acc,test_loss))


    # 对比全局模型和各簇模型的性能，验证聚类方法的有效性。
    def _compare_global_and_cluster_models(self, round_idx):
        print(f"====================== 对比全局模型和各簇模型的性能 (第 {round_idx} 轮) ======================")
        w_global = self.model_trainer.get_model_params()
        global_metrics = {"num_samples": 0, "num_correct": 0, "losses": 0}
        cluster_metrics = {cluster_id: {"num_samples": 0, "num_correct": 0, "losses": 0} for cluster_id in
                           self.cluster_models}
        # 对比全局模型在全局测试集上的表现
        print(" 1. 全局模型在全局测试集上的表现: \n")
        self._global_test_on_global_model(round_idx, w_global)
        # 对比各个簇的模型在各自簇的局部数据集上的表现
        print(" 2. 各个簇模型在局部数据集上的表现: \n")
        client = Client(
            0,
            self.train_data_local_dict[0],
            self.test_data_local_dict[0],
            self.train_data_local_num_dict[0],
            self.args,
            self.device,
            create_model_trainer(self.model, self.args),
        )
        for cluster_id, cluster_model in self.cluster_models.items():
            test_metrics = {"num_samples": [], "losses": [], "num_correct": []}
            client.model_trainer.set_model_params(cluster_model)
            client.update_local_dataset(0, None,self.test_global,None)
            test_metrics = client.local_test(True)
            print(f"簇 {cluster_id} 的准确率: {test_metrics['test_correct']/test_metrics['test_total']}, 损失: {test_metrics['test_loss']/test_metrics['test_total']}\n")
            if self.args.enable_wandb:
                wandb.log({f"Cluster-{cluster_id}Test/Acc (global)":test_metrics["test_correct"]/test_metrics["test_total"],"round":round_idx})
                wandb.log({f"Cluster-{cluster_id}Test/Loss (global)":test_metrics["test_loss"]/test_metrics["test_total"],"round":round_idx})

    # 获取客户端所属的簇 ID
    def _get_client_cluster_id(self, client_idx):
        return self.client_cluster_idx_map[client_idx]


    # 微调元模型并测试性能
    def _fine_tune_meta_model_and_test(self, meta_model, client_idx):
        client = self.client_list[client_idx]
        client.update_local_dataset(client_idx, self.train_data_local_dict[client_idx],
                                    self.test_data_local_dict[client_idx],
                                    self.train_data_local_num_dict[client.client_idx])
        fine_tune_model = client.train(copy.deepcopy(meta_model))
        # 测试 fine_tune_model
        # 测试
        client.model_trainer.set_model_params(fine_tune_model)
        test_local_metrics = client.local_test(True)
        print(f"1. 微调后的模型在Client_{client_idx}测试集上的准确率: {test_local_metrics['test_correct'] / test_local_metrics['test_total']}\n")