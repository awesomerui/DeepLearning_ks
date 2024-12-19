import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from model import AIModel, BaggingModel
from config import *
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import matplotlib.pyplot as plt

root_dir = "D:/Users/86138/Desktop/python/learn/qinghua/data"
train_dataset = AIDataset(root_dir=root_dir, transform=data_transforms["train"])
val_dataset = train_dataset
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=1)
# 记录日志的时间戳
current_time = datetime.now().strftime("%d_%H-%M-%S")
# 设置TensorBoard的日志目录
log_dir = f"D:/Users/86138/Desktop/python/learn/qinghua/tf-logs/{current_time}"
# 初始化TensorBoard写入器
writer = SummaryWriter(log_dir)
def train(model, dataloader, criterion, optimizer, device, epoch, model_index):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    for inputs, labels in tqdm(dataloader, desc=f"Training Model {model_index}"):
        inputs, labels = inputs.to(device), labels.float().unsqueeze(1).to(device)
        # 清零梯度
        optimizer.zero_grad()
        # 前向传播
        outputs = model(inputs)
        # 计算损失
        loss = criterion(outputs, labels)
        # 反向传播
        loss.backward()
        # 更新权重
        optimizer.step()
        # 累加损失
        running_loss += loss.item() * inputs.size(0)
        # 计算预测标签
        predicted_labels = (outputs > 0.5).float()
        # 统计正确预测的数量
        correct_predictions += (predicted_labels == labels).sum().item()
        # 统计总预测数量
        total_predictions += labels.size(0)
    # 计算平均损失和准确率
    epoch_loss = running_loss / len(dataloader.dataset)
    accuracy = correct_predictions / total_predictions
    # 在TensorBoard中记录训练损失和准确率
    writer.add_scalar(f"Loss/Train_Model_{model_index}", epoch_loss, epoch + 1)
    writer.add_scalar(f"Accuracy/Train_Model_{model_index}", accuracy, epoch + 1)
    print(f"Model {model_index} Training Loss: {epoch_loss:.5f}, Accuracy: {accuracy:.4f}")
    return epoch_loss, accuracy

def validate(models, dataloader, criterion, device, epoch):
    bagging_model = BaggingModel(models).to(device)
    bagging_model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.float().unsqueeze(1).to(device)
            # 前向传播
            outputs = bagging_model(inputs)
            # 计算损失
            loss = criterion(outputs, labels)
            # 累加损失
            running_loss += loss.item() * inputs.size(0)
            # 计算预测标签
            predicted_labels = (outputs > 0.5).float()
            # 统计正确预测的数量
            correct_predictions += (predicted_labels == labels).sum().item()
            # 统计总预测数量
            total_predictions += labels.size(0)
    # 计算平均损失和准确率
    epoch_loss = running_loss / len(dataloader.dataset)
    accuracy = correct_predictions / total_predictions
    # 在TensorBoard中记录验证损失和准确率
    writer.add_scalar("Loss/Validation", epoch_loss, epoch + 1)
    writer.add_scalar("Accuracy/Validation", accuracy, epoch + 1)
    print(f"Validation Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}\n")
    return bagging_model, epoch_loss, accuracy

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 创建多个AI模型实例
    models = [
        AIModel(efficientnet_type="efficientnet-b0").to(device),
        AIModel(efficientnet_type="efficientnet-b0").to(device),
        AIModel(efficientnet_type="efficientnet-b1").to(device),
        AIModel(efficientnet_type="efficientnet-b1").to(device),
        AIModel(efficientnet_type="efficientnet-b1").to(device),
    ]
    train_loaders = []
    optimizers = []
    schedulers = []
    criterions = []
    for i in range(len(models)):
        # 创建随机采样器
        sampler = RandomSampler(train_dataset, replacement=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            sampler=sampler,
            num_workers=1,
        )
        train_loaders.append(train_loader)
        # 创建优化器
        optimizer = optim.AdamW(models[i].parameters())
        # 创建学习率调度器
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2
        )
        optimizers.append(optimizer)
        schedulers.append(scheduler)
        # 创建损失函数
        criterion = nn.BCELoss()
        criterions.append(criterion)
    num_epochs = 50

    train_losses = [[] for _ in range(len(models))]
    train_accuracies = [[] for _ in range(len(models))]
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        for i, model in enumerate(models):
            epoch_loss, epoch_accuracy = train(
                model,
                train_loaders[i],
                criterions[i],
                optimizers[i],
                device,
                epoch,
                i + 1,
            )
            train_losses[i].append(epoch_loss)
            train_accuracies[i].append(epoch_accuracy)
            schedulers[i].step(epoch + epoch / len(train_loaders[i]))

        # 每个epoch结束后对验证集进行评估
        bagging_model, epoch_loss, epoch_accuracy = validate(models, val_loader, criterions[0], device, epoch)
        val_losses.append(epoch_loss)
        val_accuracies.append(epoch_accuracy)

    # 绘制训练和验证的损失及准确性的变化图
    epochs_range = list(range(1, num_epochs + 1))
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    for i in range(len(models)):
        axs[0, 0].plot(epochs_range, train_losses[i], label=f'Model {i + 1}')
    axs[0, 0].set_title('Training Loss')
    axs[0, 0].legend()

    for i in range(len(models)):
        axs[0, 1].plot(epochs_range, train_accuracies[i], label=f'Model {i + 1}')
    axs[0, 1].set_title('Training Accuracy')
    axs[0, 1].legend()

    axs[1, 0].plot(epochs_range, val_losses, label='Bagging Model')
    axs[1, 0].set_title('Validation Loss')
    axs[1, 0].legend()

    axs[1, 1].plot(epochs_range, val_accuracies, label='Bagging Model')
    axs[1, 1].set_title('Validation Accuracy')
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()

    bagging_model = BaggingModel(models).to(device)
    torch.save(bagging_model, "model.pth")
    writer.close()







