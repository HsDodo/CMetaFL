import copy
import numpy as np
from tqdm import tqdm

from .ModelTrainer import create_model_trainer
from .client import Client
import wandb
import logging
mahal_scores = {}
class FedAvg(object):
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

        self.model_trainer = create_model_trainer(model,args)
        self.model = model
        print("模型(model) = {}".format(model))
        print("设备环境(device) = {}".format(device))
        self._setup_clients(
            train_data_local_num_dict, train_data_local_dict, test_data_local_dict, self.model_trainer,
        )


    # 设置客户端,初始化客户端list
    def _setup_clients(
            self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer,
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
                model_trainer,
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
        print("训练器 = {}".format(self.model_trainer))
        w_global = self.model_trainer.get_model_params()
        for round_idx in tqdm(range(self.args.comm_round),ncols=80,desc="通信轮数",leave=True):
            print("############# 第{}轮通信(Start) #############".format(round_idx))
            w_locals = []
            client_indexes = self._client_sampling(
                round_idx, self.args.client_num_in_total, self.args.client_num_per_round
            )
            print("客户端id = %s" % str(client_indexes))
            for idx,client in enumerate(self.client_list):
                client_idx = client_indexes[idx]
                client.update_local_dataset(
                    client_idx,
                    self.train_data_local_dict[client_idx],
                    self.test_data_local_dict[client_idx],
                    self.train_data_local_num_dict[client_idx]
                )

                # 根据客户端的模型训练器进行模型训练
                w = client.train(copy.deepcopy(w_global))
                w_locals.append((client.get_sample_number(), copy.deepcopy(w)))
            w_global = self._aggregate(w_locals,w_global,round_idx)
            self.model_trainer.set_model_params(w_global)
            self._global_test_on_global_model(round_idx,w_global)

    # 加权聚合函数
    def _aggregate(self, w_locals,w_globals,round_idx):
        client_attack_idx = []
        # 开启防御
        adjusted_weight =[]
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, averaged_params) = w_locals[idx]
            training_num += sample_num
        (sample_num, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            for i in range(len(w_locals)):
                local_sample_num, local_params = w_locals[i]
                if len(adjusted_weight) != 0:
                    weight = (local_sample_num / training_num) * adjusted_weight[i]
                else:
                    weight = (local_sample_num / training_num)
                if i == 0:
                    averaged_params[k] = local_params[k] * weight
                else:
                    averaged_params[k] += local_params[k] * weight
        return averaged_params

    # 平均聚合函数，没有加权
    def _aggregate_noniid_avg(self,w_locals):
        (_, averaged_params) = w_locals[0]
        for k in averaged_params.keys:
            for idx in range(0,len(w_locals)):
                (_, local_params) = w_locals[idx]
                averaged_params[k] += local_params[k]
            averaged_params[k] /= len(w_locals)
        return averaged_params

    # # 生成验证集
    # def _generate_validation_set(self,num_samples=10000):
    #         test_data_num = len(self.test_global.dataset)
    #         sample_indices = random.sample(range(test_data_num), min(test_data_num,num_samples)) # 从测试集中随机抽取10000个样本
    #         subset = torch.utils.data.Subset(self.test_global.dataset, sample_indices)
    #         sample_testset = torch.utils.data.DataLoader(subset, batch_size=self.args.batch_size)
    #         self.val_global = sample_testset

    def _local_test_on_all_clients(self,round_idx):
        print("############# 在所有客户端上进行评估 (第 {} 轮) #############".format(round_idx))
        train_metrics = {"num_samples": [], "losses": [], "num_correct": []}
        test_metrics = {"num_samples": [], "losses": [], "num_correct": []}

        client = self.client_list[0]
        for client_idx in tqdm(range(self.args.client_num_in_total),ncols=80,desc="在所有客户端上验证",leave=True):
            if self.test_data_local_dict[client_idx] is None:
                continue
            client.update_local_dataset(
                0,
                self.train_data_local_dict[client_idx],
                self.test_data_local_dict[client_idx],
                self.train_data_local_num_dict[client_idx],
            )
            #训练
            train_local_metrics = client.local_test(False) # 训练集上评估,
            train_metrics["num_samples"].append(copy.deepcopy(train_local_metrics["test_total"]))
            train_metrics["num_correct"].append(copy.deepcopy(train_local_metrics["test_correct"]))
            train_metrics["losses"].append(copy.deepcopy(train_local_metrics["test_loss"]))

            #测试
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

        stats = {"train_acc": train_acc, "train_loss": train_loss }
        if self.args.enable_wandb:
            wandb.log({"Train/Acc":train_acc,"round":round_idx})
            wandb.log({"Train/Loss":train_loss,"round":round_idx})
        print(stats)


        stats = {"test_acc": test_acc, "test_loss": test_loss }
        if self.args.enable_wandb:
            wandb.log({"Test/Acc":test_acc,"round":round_idx})
            wandb.log({"Test/Loss":test_loss,"round":round_idx})
        print(stats)


    def _global_test_on_global_model(self,round_idx,w_global):
        print("############# 在全局模型上进行评估 (第 {} 轮) #############".format(round_idx))
        client = self.client_list[0]
        client.update_local_dataset(0,None,self.test_global,None)  # 设置全局测试集
        client.model_trainer.set_model_params(w_global)        # 设置全局模型
        test_metrics = client.local_test(True)           # 在全局测试集上进行评估
        if self.args.enable_wandb:
            wandb.log({"Test/Acc (global)":test_metrics["test_correct"]/test_metrics["test_total"],"round":round_idx})
            wandb.log({"Test/Loss (global)":test_metrics["test_loss"]/test_metrics["test_total"],"round":round_idx})

    # def _local_test_on_validation_set(self,round_idx):
    #     print("############# 在验证集上进行评估 (第 {} 轮) #############".format(round_idx))
    #     if self.val_global is None:
    #         self._generate_validation_set()
    #
    #     client = self.client_list[0]
    #     client.update_local_dataset(0,None,self.val_global,None)
    #
    #     test_metrics = client.local_test(True)
    #
    #     if self.args.dataset == "stackoverflow_nwp":
    #         test_acc = test_metrics["test_correct"] / test_metrics["test_total"]
    #         test_loss = test_metrics["test_loss"] / test_metrics["test_total"]
    #         stats = {"test_acc": test_acc, "test_loss": test_loss}
    #         if self.args.enable_wandb:
    #             wandb.log({"Test/Acc": test_acc, "round": round_idx})
    #             wandb.log({"Test/Loss": test_loss, "round": round_idx})
    #     elif self.args.dataset == "stackoverflow_lr":
    #         test_acc = test_metrics["test_correct"] / test_metrics["test_total"]
    #         test_pre = test_metrics["test_precision"] / test_metrics["test_total"]
    #         test_rec = test_metrics["test_recall"] / test_metrics["test_total"]
    #         test_loss = test_metrics["test_loss"] / test_metrics["test_total"]
    #         stats = {
    #             "test_acc": test_acc,
    #             "test_pre": test_pre,
    #             "test_rec": test_rec,
    #             "test_loss": test_loss,
    #         }
    #         if self.args.enable_wandb:
    #             wandb.log({"Test/Acc": test_acc, "round": round_idx})
    #             wandb.log({"Test/Pre": test_pre, "round": round_idx})
    #             wandb.log({"Test/Rec": test_rec, "round": round_idx})
    #             wandb.log({"Test/Loss": test_loss, "round": round_idx})
    #     else:
    #         raise Exception("Unknown format to log metrics for dataset {}!" % self.args.dataset)
    #     print(stats)
