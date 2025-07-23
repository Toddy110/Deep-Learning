# MNIST 多层感知机（MLP）分类器代码详解

本项目实现了一个基于 PyTorch 的多层感知机（MLP）模型，用于对 MNIST 手写数字图片进行分类。本文将对 `mnist_mlp_classifier.py` 代码进行逐行详细解析，帮助初学者理解每一步的设计思路、参数设置原因，以及涉及到的 PyTorch 和 torchvision 库函数的用法。

---

## 目录
1. [数据预处理与加载](#数据预处理与加载)
2. [神经网络结构定义](#神经网络结构定义)
3. [损失函数与优化器](#损失函数与优化器)
4. [训练函数详解](#训练函数详解)
5. [测试函数详解](#测试函数详解)
6. [主程序入口](#主程序入口)
7. [参数设置说明](#参数设置说明)
8. [PyTorch/torchvision 相关API详解](#pytorchtorchvision-相关api详解)
9. [小结](#小结)

---

## 数据预处理与加载

### 1.1 导入相关库
```python
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
```
- `torch`：PyTorch 的核心库，包含张量、自动求导、神经网络等模块。
- `torchvision.transforms`：常用的数据预处理方法集合。
- `torchvision.datasets`：常用数据集的下载与加载接口。
- `torch.utils.data.DataLoader`：用于批量加载数据，支持多线程和数据打乱。
- `torch.nn.functional`：包含常用的无状态神经网络函数（如激活函数 ReLU）。
- `torch.optim`：优化器模块，包含 SGD、Adam 等。

### 1.2 批量大小设置
```python
batch_size = 64
```
- **为什么设置为64？**
  - 批量大小（batch size）是深度学习训练中的重要超参数。较小的 batch size（如32、64）可以加快模型收敛速度，提升泛化能力，同时不会占用过多内存。64 是常用的折中选择。

### 1.3 数据预处理操作
```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
```
- `transforms.Compose`：将多个预处理操作串联起来。
- `transforms.ToTensor()`：将 PIL 图片或 numpy 数组转为 PyTorch 的 FloatTensor，并自动归一化到 [0,1]。
- `transforms.Normalize((0.1307,), (0.3081,))`：对每个像素做标准化，`(x-mean)/std`，其中 mean=0.1307，std=0.3081 是 MNIST 全体像素的统计值。
  - **为什么要标准化？**
    - 标准化可以加快模型收敛速度，提升训练稳定性。

### 1.4 加载数据集
```python
train_dataset = datasets.MNIST(
    root='D:/PythonCode/Pytorch_learning/MNIST',
    train=True,
    download=True,
    transform=transform
)
```
- `root`：数据集存放路径。
- `train=True`：加载训练集。
- `download=True`：本地没有数据时自动下载。
- `transform`：应用上面定义的预处理。

同理，测试集：
```python
test_dataset = datasets.MNIST(
    root='D:/PythonCode/Pytorch_learning/MNIST',
    train=False,
    download=True,
    transform=transform
)
```

### 1.5 构建数据加载器
```python
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)
```
- `DataLoader`：将数据集分批次加载，支持多线程和数据打乱。
- `shuffle=True`：训练集每个 epoch 前打乱，提升泛化能力。
- `shuffle=False`：测试集不打乱，保证评估一致性。

---

## 神经网络结构定义

### 2.1 定义多层感知机（MLP）
```python
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(784, 512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = self.l5(x)
        return x
```
- 继承自 `torch.nn.Module`，这是所有神经网络模块的基类。
- `__init__` 方法中定义了5个全连接层（Linear）：
  - `self.l1 = torch.nn.Linear(784, 512)`：输入层，28x28=784个像素，输出512个特征。
  - `self.l2 = torch.nn.Linear(512, 256)`：第1隐藏层。
  - `self.l3 = torch.nn.Linear(256, 128)`：第2隐藏层。
  - `self.l4 = torch.nn.Linear(128, 64)`：第3隐藏层。
  - `self.l5 = torch.nn.Linear(64, 10)`：输出层，10个类别。
- `forward` 方法定义前向传播：
  - `x.view(-1, 784)`：将输入图片展平成一维向量。
  - 每层后接 ReLU 激活函数（`F.relu`），输出层不加激活，直接输出 logits。

#### **参数设置理由**
- 多层结构（深度）有助于模型学习更复杂的特征。
- 隐藏层神经元数量逐层递减，有助于特征压缩和泛化。
- 输出层为10，对应数字0-9。

---

## 损失函数与优化器

### 3.1 损失函数
```python
criterion = torch.nn.CrossEntropyLoss()
```
- `CrossEntropyLoss`：适用于多分类任务，自动将 logits 通过 softmax 归一化，并计算交叉熵损失。

### 3.2 优化器
```python
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
```
- `optim.SGD`：随机梯度下降优化器。
- `model.parameters()`：需要优化的参数。
- `lr=0.01`：学习率，控制每次参数更新的步长。
- `momentum=0.5`：动量项，有助于加速收敛，减少震荡。

#### **参数设置理由**
- 学习率0.01是常用的初始值，适合大多数场景。
- 动量0.5可以在一定程度上提升训练速度和稳定性。

---

## 训练函数详解

```python
def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0
```
- `train(epoch)`：训练模型一个 epoch。
- `optimizer.zero_grad()`：每次反向传播前清空梯度，防止梯度累加。
- `outputs = model(inputs)`：前向传播，得到预测结果。
- `loss = criterion(outputs, target)`：计算损失。
- `loss.backward()`：反向传播，计算梯度。
- `optimizer.step()`：更新参数。
- `running_loss`：累计损失，每300个 batch 打印一次平均损失。

#### **参数说明**
- `epoch`：当前训练轮数。
- `batch_idx`：当前 batch 的编号。

---

## 测试函数详解

```python
def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
```
- `torch.no_grad()`：测试时不计算梯度，节省内存和加快速度。
- `torch.max(outputs.data, dim=1)`：返回每行最大值和对应的索引，索引即为预测类别。
- 统计预测正确的样本数，计算准确率。

#### **参数说明**
- `outputs.data`：模型输出的 logits。
- `dim=1`：在类别维度上取最大值。

---

## 主程序入口

```python
if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()
```
- 训练和测试共进行10个 epoch。
- 每个 epoch 后在测试集上评估模型性能。

#### **参数说明**
- `range(10)`：训练10轮。

---

## 参数设置说明

- **batch_size=64**：平衡训练速度和内存消耗。
- **学习率 lr=0.01**：适合大多数场景的初始值。
- **momentum=0.5**：加速收敛，减少震荡。
- **隐藏层神经元数**：逐层递减，便于特征压缩和泛化。
- **epoch=10**：适合初学者快速观察模型收敛情况。

---

## PyTorch/torchvision 相关API详解

- `torch.nn.Module`：所有神经网络模块的基类。
- `torch.nn.Linear(in_features, out_features)`：全连接层。
- `torch.nn.functional.relu(x)`：ReLU 激活函数。
- `torch.utils.data.DataLoader(dataset, batch_size, shuffle)`：批量加载数据。
- `torchvision.datasets.MNIST`：自动下载和加载 MNIST 数据集。
- `torchvision.transforms.Compose([...])`：组合多个预处理操作。
- `torchvision.transforms.ToTensor()`：图片转为张量。
- `torchvision.transforms.Normalize(mean, std)`：标准化。
- `torch.optim.SGD(params, lr, momentum)`：SGD 优化器。
- `torch.nn.CrossEntropyLoss()`：交叉熵损失。
- `torch.no_grad()`：上下文管理器，禁用梯度计算。
- `torch.max(input, dim)`：返回最大值和索引。

---

## 小结

本项目完整演示了用 PyTorch 实现多层感知机（MLP）进行 MNIST 手写数字识别的全过程，包括数据预处理、模型构建、训练与测试。结构清晰，注释详细，非常适合初学者学习和参考。
