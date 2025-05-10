import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model.models import LSTMModel


# 客户端本地训练（Reptile方法）
def reptile_train(model, client_loader, inner_steps, inner_lr, criterion, pairwise_loss, margin=1.0):
    model.train()
    original_state = {k: v.clone() for k, v in model.state_dict().items()}
    optimizer = optim.SGD(model.parameters(), lr=inner_lr)

    for _ in range(inner_steps):
        for data, target in client_loader:
            data, target = data.to(model.fc.weight.device), target.to(model.fc.weight.device)
            optimizer.zero_grad()
            output = model(data)
            classification_loss = criterion(output, target)

            # Pairwise similarity loss
            embeddings = model.get_embedding(data)
            pairwise_targets = torch.randint(0, 2, (embeddings.size(0),))  # Random pairs
            pairwise_loss_value = pairwise_loss(embeddings, embeddings[torch.randperm(embeddings.size(0))], pairwise_targets)

            loss = classification_loss + pairwise_loss_value
            loss.backward()
            optimizer.step()

    updated_state = model.state_dict()
    return {k: original_state[k] + (updated_state[k] - original_state[k]) / inner_steps for k in original_state.keys()}

# 对比损失函数
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive



# 加载对应数据集
def generate_client_data(args, num_clients, num_samples_per_client, input_size, sequence_length, num_classes):
    client_data = []
    #  加载数据集 待写
    # UCI-HAR

    # PAMAP2

    # WISDM

    # OPPORTUNITY

    return client_data



# 服务器端模型聚合
def server_aggregate(global_weights, client_weights, learning_rate):
    new_weights = {}
    for key in global_weights.keys():
        new_weights[key] = global_weights[key] + learning_rate * (torch.mean(torch.stack([client_weight[key] for client_weight in client_weights]), dim=0) - global_weights[key])
    return new_weights


# FedReptile算法
def fedreptile(num_clients, num_samples_per_client, num_rounds, inner_steps, inner_lr, learning_rate, input_size, sequence_length, num_classes):
    global_model = LSTMModel(input_size, hidden_size=64, num_layers=2, num_classes=num_classes).cuda()
    global_weights = global_model.state_dict()
    ## 获取客户端数据
    client_data = generate_client_data(num_clients, num_samples_per_client, input_size, sequence_length, num_classes)
    criterion = nn.CrossEntropyLoss()
    pairwise_loss = ContrastiveLoss(margin=1.0)

    for round in range(num_rounds):
        client_weights = []
        for client_loader in client_data:
            client_model = LSTMModel(input_size, hidden_size=64, num_layers=2, num_classes=num_classes).cuda()
            client_model.load_state_dict(global_weights)
            new_weights = reptile_train(client_model, client_loader, inner_steps, inner_lr, criterion, pairwise_loss)
            client_weights.append(new_weights)
        global_weights = server_aggregate(global_weights, client_weights, learning_rate)
        global_model.load_state_dict(global_weights)
        print(f'通信轮次: {round + 1} 完成.')
    return global_model


