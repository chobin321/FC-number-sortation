import torch
import matplotlib.pyplot as plt
# import torchvision
# import torchaudio

from torch import nn
from torch.utils.data import DataLoader #batch
from torchvision import datasets #MNIST
from torchvision.transforms import ToTensor #格式转换

# 引入我本地MNIST数据集中的训练数据集
training_data = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=ToTensor()
)
print(len(training_data)) # 60000

# 引入我本地MNIST数据集中的测试数据集
test_data = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=ToTensor()
)
print(len(test_data)) # 10000

# 创建dataloader将数据集分割为batch
train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 构建卷积神经网络 (CNN)
class CNNNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. 第一个卷积块 
        # 输入通道 1 (灰度图)，输出通道 32 ，卷积核大小 3x3，步长 1，填充 1 (保持图像大小 28x28 不变)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        # 最大池化，窗口 2x2，步长 2 (把 28x28 的图像变成 14x14)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 2. 第二个卷积块 
        # 输入通道 32，输出通道 64，卷积核 3x3
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        # 再次池化 (把 14x14 变成 7x7)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 3. 全连接层 (把提取到的特征汇总分类)
        self.flatten = nn.Flatten()
        # 经过两次池化后，图像变成了 64 个通道，每个通道 7x7，所以展平后长度是 64 * 7 * 7 = 3136
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=128, out_features=10) # 最后输出 10 个数字(0-9)

    # 前向传播 (数据流向)
    def forward(self, x):
        # 第一层卷积
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # 第二层卷积
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # 全连接层
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

# 定义训练函数
def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train() # 训练模式
    total_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # 计算预测误差
        pred = model(X)
        loss = loss_fn(pred, y)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch % 100 == 0:
            current = batch * len(X)
            print(f"loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")
    return total_loss / len(dataloader) # 返回平均损失

# 定义测试函数
def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval() # 测试模式
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, correct*100


if __name__ == "__main__":

    # 选择运行设备
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using {device} device")

    # 实例化模型
    model = CNNNetwork().to(device)
    print(model) # 打印网络结构

    loss_fn = nn.CrossEntropyLoss() # 交叉熵损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) # Adam优化器

    epochs = 3 # 把整个数据集完整过 3 遍
    # 建立空列表以存储用于绘制训练损失和测试准确率的图表的数据
    train_losses = []
    test_accs = []
    test_losses = []
    for t in range(epochs):
        print(f"-------------------------------\nEpoch {t+1}\n-------------------------------")
        
        # 调用训练函数
        train_loss = train(train_dataloader, model, loss_fn, optimizer, device)
        train_losses.append(train_loss)
        
        # 调用测试函数
        test_loss, test_acc = test(test_dataloader, model, loss_fn, device)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

    print("Done!")

    # 绘制训练损失和测试准确率的图表
    # 1. 建一块大画布，宽 12，高 5
    plt.figure(figsize=(12, 5))

    # 2. 画左边的图 (Loss曲线)
    # 1行2列，第1个位置
    plt.subplot(1, 2, 1) 
    # 画折线：X轴是 epoch 序号 (1到3)，Y轴是 train_losses，设置个标记点和名字
    plt.plot(range(1, epochs + 1), train_losses, marker='o', label='Train Loss')
    # 测试集的 loss 
    plt.plot(range(1, epochs + 1), test_losses, marker='s', label='Test Loss')
    plt.title('Loss Curve') # 标题
    plt.xlabel('Epoch') # X轴名字
    plt.ylabel('Loss')  # Y轴名字
    plt.legend() # 把 label 显示出来（图例）

    # 3. 画右边的图 (Accuracy曲线)
    # 1行2列，第2个位置
    plt.subplot(1, 2, 2) 
    plt.plot(range(1, epochs + 1), test_accs, marker='o', color='green', label='Test Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    # 4. 显示图表
    plt.show()