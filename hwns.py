import torch
import torchvision
# import torchaudio

# 环境版本验证
# print(torch.__version__)
# print(torchvision.__version__)
# print(torchaudio.__version__)

from torch import nn
from torch.utils.data import DataLoader #batch
from torchvision import datasets #MNIST
from torchvision.transforms import ToTensor #格式转换

# 引入我本地MNIST数据集中的训练数据集
training_data = datasets.MNIST(
    root="C:/Users/chobin/Desktop/vision/dataset",
    train=True,
    download=True,
    transform=ToTensor()
)
print(len(training_data)) # 60000

# 引入我本地MNIST数据集中的测试数据集
test_data = datasets.MNIST(
    root="C:/Users/chobin/Desktop/vision/dataset",
    train=False,
    download=True,
    transform=ToTensor()
)
print(len(test_data)) # 10000

# 创建dataloader将数据集分割为batch
train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 选择运行设备
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

# 构建神经网络
class NeuralNetwork(nn.Module):
    # 定义全连接神经网络结构
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.hidden1 = nn.Linear(in_features=28*28, out_features=128)
        self.hidden2 = nn.Linear(in_features=128, out_features=256)
        self.output = nn.Linear(in_features=256, out_features=10)

    # 前向传播
    def forward(self, x):
        x = self.flatten(x)
        x = self.hidden1(x)
        x = torch.relu(x)
        x = self.hidden2(x)
        x = torch.relu(x)
        x = self.output(x)
        return x

model = NeuralNetwork().to(device)
print(model) # 打印网络结构

# 定义训练函数
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train() # 训练模式
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # 计算预测误差
        pred = model(X)
        loss = loss_fn(pred, y)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


loss_fn = nn.CrossEntropyLoss() # 交叉熵损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) # Adam优化器

train(train_dataloader, model, loss_fn, optimizer) # 训练模型