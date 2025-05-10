import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Client:
    def __init__(
        self, client_idx, local_training_data, local_test_data, local_sample_number, args, device, model_trainer,
    ):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        self.args = args
        self.device = device
        self.model_trainer = model_trainer
        self.cluster_idx = 0



    def update_local_dataset(self, client_idx, local_training_data, local_test_data, local_sample_number):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        self.model_trainer.set_id(client_idx)

    def get_sample_number(self):
        return self.local_sample_number

    #自己添加的
    def get_client_idx(self):
        return self.client_idx



    def train(self, w_global):
        self.model_trainer.set_model_params(w_global)

        self.model_trainer.train(self.local_training_data, self.device, self.args)
        weights = self.model_trainer.get_model_params()

        return weights


    def train_finetune(self, w_global):
        self.model_trainer.set_model_params(w_global)
        self.model_trainer.train_finetune(self.local_training_data, self.device, self.args)
        weights = self.model_trainer.get_model_params()
        return weights

    def train_reptile(self, w_global):
        self.model_trainer.set_model_params(w_global)
        self.model_trainer.train_reptile(self.local_training_data, self.device, self.args)
        weights = self.model_trainer.get_model_params()

        return weights





    def local_test(self, b_use_test_dataset):
        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics

    #
    # def train_reptile(self, w_global, adaptation_steps=20):
    #     """
    #     Reptile 内循环训练：通过支持集执行多步梯度更新
    #     返回更新后的模型参数。
    #     """
    #     # 设置初始全局模型参数
    #     self.model_trainer.set_model_params(w_global)
    #
    #     model = self.model_trainer.model
    #     model.to(self.device)
    #     model.train()
    #
    #     # 定义损失函数和优化器
    #     criterion = nn.CrossEntropyLoss().to(self.device)
    #     optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=self.args.learning_rate)
    #
    #     # 保存初始模型参数
    #     original_params = {name: param.clone() for name, param in model.named_parameters()}
    #
    #     # 执行多步内循环更新
    #     for step in range(adaptation_steps):
    #         for batch_idx, (x, labels) in enumerate(self.local_training_data):
    #             x, labels = x.to(self.device), labels.to(self.device)
    #             optimizer.zero_grad()
    #             outputs = model(x)
    #             loss = criterion(outputs, labels)
    #             loss.backward()
    #             optimizer.step()
    #               # 每个 step 使用一个 batch 更新
    #
    #     # 获取更新后的模型参数
    #     weights = {name: param.clone() for name, param in model.named_parameters()}
    #
    #     # 重置模型为初始参数
    #     for name, param in model.named_parameters():
    #         param.data.copy_(original_params[name].data)
    #
    #     return weights