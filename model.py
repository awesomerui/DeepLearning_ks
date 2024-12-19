import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class AIModel(nn.Module):
    def __init__(self, efficientnet_type="efficientnet-b0"):
        super(AIModel, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained(
            efficientnet_type,  #选择的EfficientNet版本
            num_classes=1
        )
        # 使用Sigmoid激活函数
        self.sigmoid = nn.Sigmoid()

    # 定义前向传播函数
    def forward(self, x):
        # 通过EfficientNet提取特征
        x = self.efficientnet(x)
        # 将EfficientNet的输出通过Sigmoid激活函数映射到[0, 1]范围
        x = self.sigmoid(x)
        return x

class BaggingModel(nn.Module):
    def __init__(self, models):
        super(BaggingModel, self).__init__()
        self.models = nn.ModuleList(models)

    # 定义前向传播函数
    def forward(self, x):
        # 用于存储每个模型的输出
        outputs = []
        # 遍历所有模型
        for model in self.models:
            # 获取每个模型的预测输出，并添加到outputs列表中
            outputs.append(model(x))
        # 将所有模型的输出堆叠成一个张量，dim=0表示按第一个维度堆叠
        outputs = torch.stack(outputs, dim=0)
        # 对所有模型的输出取平均值
        output = torch.mean(outputs, dim=0)
        return output
