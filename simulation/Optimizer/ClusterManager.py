# ClusterManager.py
import torch
import wandb
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans, KMeans
import numpy as np
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
import torch.nn.functional as F
import seaborn as sns  # 导入 seaborn 库



class ClusterManager:
    def __init__(self,args , num_clusters, pca_components=50, use_minibatch_kmeans=True, random_state=42):
        self.num_clusters = num_clusters
        self.args = args
        self.pca = PCA(n_components=pca_components)
        self.use_minibatch_kmeans = use_minibatch_kmeans
        if use_minibatch_kmeans:
            self.kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=random_state, batch_size=10)
        else:
            self.kmeans = KMeans(n_clusters=num_clusters, random_state=random_state)

    def cluster(self, model_params):
        """
        对模型参数进行聚类。

        Args:
            model_params (np.ndarray): 降维后的模型参数矩阵，形状为 (num_clients, n_components)。

        Returns:
            np.ndarray: 每个客户端所属的簇编号。
        """
        # 聚类
        clusters = self.kmeans.fit_predict(model_params)
        return clusters


    #  计算每个客户端模型参数的余弦相似度矩阵
    def _calculate_cosine_similarity(self, w_locals):
        model_params = []
        for _, averaged_params in w_locals:
            param_vector = torch.cat([param.detach().cpu().view(-1) for param in averaged_params.values()])
            model_params.append(param_vector)
        # 构建一个矩阵，大小为 (num_clients, num_params)
        param_matrix = torch.stack(model_params)  # (num_clients, num_params)
        # 计算余弦相似度
        cosine_sim_matrix = F.cosine_similarity(param_matrix.unsqueeze(1), param_matrix.unsqueeze(0), dim=2).numpy()
        return cosine_sim_matrix


    # 将余弦相似度转化为距离矩阵
    def _calculate_distance_matrix(self, cosine_sim_matrix):
        # 距离矩阵 D = 1 - 相似度矩阵 S
        distance_matrix = 1 - cosine_sim_matrix
        np.fill_diagonal(distance_matrix, 0)  # 确保对角线为零
        return distance_matrix



    # 进行聚类并返回聚完类之后的簇编号，根据客户端的模型参数进行聚类
    def _cluster_clients(self, w_locals, round_idx):
        """
        进行聚类并返回聚完类之后的簇编号，根据客户端的模型参数进行聚类
        这次使用余弦相似度进行聚类。
        """
        # 提取每个客户端的模型参数向量
        model_params = []
        for _, params in w_locals:
            param_vector = np.concatenate([param.detach().cpu().flatten().numpy() for param in params.values()])
            model_params.append(param_vector)

        # 计算每个客户端模型参数的余弦相似度矩阵
        cosine_sim_matrix = self._calculate_cosine_similarity(w_locals)

        # 将余弦相似度转化为距离矩阵
        distance_matrix = self._calculate_distance_matrix(cosine_sim_matrix)

        # 使用 KMeans 或 MiniBatchKMeans 进行聚类
        # 注意：KMeans 需要一个二维数组作为输入，这里我们将距离矩阵转换为适合 KMeans 的输入
        # 我们使用 KMeans 聚类（可以选择 MiniBatchKMeans 作为优化）
        clusters = self.kmeans.fit_predict(distance_matrix)

        # 打印和评估聚类结果
        self.print_clusters(clusters, round_idx)
        print("聚类簇编号Clusters", clusters)
        # 评估聚类效果
        self._evaluate_clustering(distance_matrix, clusters)

        # 可视化聚类结果
        self.visualize_distance_matrix(distance_matrix, clusters, round_idx)

        return clusters

    def _adjust_clustering_parameters(self, silhouette):
        # 示例：根据轮廓系数调整簇数量
        if silhouette < self.args.silhouette_threshold and self.num_clusters < self.args.max_clusters:
            self.num_clusters += 1
            self.kmeans.n_clusters = self.num_clusters
            print(f"轮廓系数较低，增加簇数量到 {self.num_clusters}")
        elif silhouette > self.args.silhouette_threshold_high and self.num_clusters > self.args.min_clusters:
            self.num_clusters -= 1
            self.kmeans.n_clusters = self.num_clusters
            print(f"轮廓系数较高，减少簇数量到 {self.num_clusters}")



    # 根据热力图来可视化距离矩阵，可视化聚类效果
    def visualize_distance_matrix(self, distance_matrix, clusters, round_idx):
        """
        可视化基于余弦相似度的距离矩阵
        """

        # 创建一个新的图形
        plt.figure(figsize=(10, 8))

        # 使用 seaborn 的 heatmap 绘制热力图
        # sns.set(style="whitegrid")  # 设置背景网格
        ax = sns.heatmap(distance_matrix, cmap='Blues', square=True, annot=False, fmt='g', linewidths=0.5)

        # 添加标题和标签
        plt.title(f"Distance Matrix at Round {round_idx}", fontsize=16)
        plt.xlabel("Clients", fontsize=14)
        plt.ylabel("Clients", fontsize=14)

        # 添加簇的分隔线，用于区分不同簇的客户端
        for i in range(self.num_clusters):
            indices = [idx for idx, label in enumerate(clusters) if label == i]
            ax.add_patch(plt.Rectangle((min(indices), min(indices)), len(indices), len(indices), fill=False, edgecolor='red', lw=3))
        # 显示热力图
        plt.savefig(f"distance_matrix_round_{round_idx}.png", dpi=300, bbox_inches='tight')
        print("保存热力图成功")



    def visualize_clusters(self, model_params, clusters, round_idx, sns=None):
        """
        可视化聚类结果
        """
        try:
            # 使用 PCA 进行降维，以便在二维平面上可视化
            pca = PCA(n_components=2)
            reduced_params = pca.fit_transform(model_params)

            # 创建一个新的图形
            plt.figure(figsize=(12, 8))
            sns.set(style="whitegrid")  # 设置背景网格

            # 使用 seaborn 的 scatterplot 绘制散点图
            scatter = sns.scatterplot(x=reduced_params[:, 0], y=reduced_params[:, 1], hue=clusters, palette='viridis', s=100, edgecolor='k', alpha=0.7)

            # 添加标题和标签
            plt.title(f"Client Clusters at Round {round_idx}", fontsize=16)
            plt.xlabel("Principal Component 1", fontsize=14)
            plt.ylabel("Principal Component 2", fontsize=14)

            # 添加图例
            plt.legend(title="Clusters", fontsize=12, title_fontsize=14)

            # 保存图形
            plt.savefig(f"clusters_round_{round_idx}.png", dpi=300, bbox_inches='tight')

            # 显示图形
            plt.show()

        except Exception as e:
            print(f"聚类可视化失败: {e}")

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

    def _evaluate_clustering(self, distance_matrix, clusters):
        """
        评估聚类结果的效果
        """
        try:
            silhouette = silhouette_score(distance_matrix, clusters, metric='cosine')
            ch_score = calinski_harabasz_score(distance_matrix, clusters)
            db_score = davies_bouldin_score(distance_matrix, clusters)
            wandb.log({"silhouette_score": silhouette, "calinski_harabasz_score": ch_score, "davies_bouldin_score": db_score})
            print(f"Silhouette Score: {silhouette}")
            print(f"Calinski-Harabasz Score: {ch_score}")
            print(f"Davies-Bouldin Score: {db_score}")
            # 动态调整聚类参数
            self._adjust_clustering_parameters(silhouette)
        except Exception as e:
            print(f"聚类评估失败: {e}")