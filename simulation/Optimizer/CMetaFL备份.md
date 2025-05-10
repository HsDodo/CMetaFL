



1.16 计算每个簇内的簇平均性能

```python
   def _compare_global_and_cluster_models(self, round_idx):
        print(f"====================== 对比全局模型和各簇模型的性能 (第 {round_idx} 轮) ======================")
        w_global = self.model_trainer.get_model_params()
        # 对比全局模型在全局测试集上的表现
        print(" 1. 全局模型在全局测试集上的表现: \n")
        self._global_test_on_global_model(round_idx, w_global)
        # 对比各个簇的模型在各自簇的局部数据集上的表现
        print(" 2. 簇模型的平均表现: \n")
        client = Client(
            0,
            self.train_data_local_dict[0],
            self.test_data_local_dict[0],
            self.train_data_local_num_dict[0],
            self.args,
            self.device,
            create_model_trainer(self.model, self.args),
        )
        cluster_metrics = {"accuracy": [], "loss": []}
        for cluster_id, cluster_model in self.cluster_models.items():
            client.model_trainer.set_model_params(cluster_model)
            # 计算簇 cluster_id 内的总体性能
            num_client = 0
            cluster_client_metrics = {"accuracy": [], "loss": []}
            for client_idx in range (self.args.client_num_in_total):
                if self.client_cluster_idx_map[client_idx] == cluster_id:
                    num_client += 1
                    client.update_local_dataset(
                        client_idx,
                        self.train_data_local_dict[client_idx],
                        self.train_data_local_dict[client_idx],
                        self.train_data_local_num_dict[client_idx],
                    )

                    finetune_model = client.train_finetune(copy.deepcopy(cluster_model))
                    client.model_trainer.set_model_params(finetune_model)
                    test_metrics = client.local_test(True)

                    # 记录簇 cluster_id 内每个客户端的性能
                    cluster_client_metrics["accuracy"].append(test_metrics["test_correct"] / test_metrics["test_total"])
                    cluster_client_metrics["loss"].append(test_metrics["test_loss"] / test_metrics["test_total"])
                    print("簇 {} 客户端 {} 测试集上的准确率: {:.4f}, 损失: {:.4f}".format(cluster_id, client_idx,
                                                                                   test_metrics["test_correct"] / test_metrics["test_total"],
                                                                                   test_metrics["test_loss"] / test_metrics["test_total"]))
            # 记录簇 cluster_id 内的总体平均性能
            if num_client > 0:
                print("簇 {} 内的客户端数: {} 平均准确率: {:.4f}, 平均损失: {:.4f}".format(cluster_id, num_client,
                                                                                   sum(cluster_client_metrics["accuracy"]) / num_client,
                                                                                   sum(cluster_client_metrics["loss"]) / num_client))
                cluster_metrics["accuracy"].append(sum(cluster_client_metrics["accuracy"]) / num_client)
                cluster_metrics["loss"].append(sum(cluster_client_metrics["loss"]) / num_client)
        if len(cluster_metrics["accuracy"]) == 0 or len(cluster_metrics["loss"]) == 0:
            print("无法计算簇模型的平均性能，因为没有客户端被分配到任何簇。\n")
            return
        if self.args.enable_wandb:
            # 记录簇模型的平均性能 Average Performance of Cluster Models
                wandb.log({f"Cluster Test/Acc (Avg)": sum(cluster_metrics["accuracy"]) / len(cluster_metrics["accuracy"]), "round": round_idx})
                wandb.log({f"Cluster Test/Loss (Avg)":sum(cluster_metrics["loss"]) / len(cluster_metrics),"round":round_idx})
        avg_acc = sum(cluster_metrics["accuracy"]) / len(cluster_metrics["accuracy"])
        avg_loss = sum(cluster_metrics["loss"]) / len(cluster_metrics["loss"])
        print(f"cluster Avg Acc: {avg_acc}, Avg Loss: {avg_loss}\n")
```