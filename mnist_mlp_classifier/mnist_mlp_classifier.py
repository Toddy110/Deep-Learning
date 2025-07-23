import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader  # 用于构建数据加载器，将数据集分批次加载
import torch.nn.functional as F  # 用于调用常用的神经网络函数（如ReLU激活函数）
import torch.optim as optim  # 用于构建优化器（如SGD）

# =========================
# 1. 数据预处理与加载
# =========================

batch_size = 64  # 每个批次加载64张图片（mini-batch大小）

# 定义数据预处理操作
# transforms.Compose：将多个预处理操作组合在一起
# transforms.ToTensor()：将PIL图片或numpy.ndarray转换为torch.FloatTensor，并归一化到[0,1]
# transforms.Normalize((0.1307,), (0.3081,))：对每个像素进行标准化，(x-mean)/std
# 0.1307为MNIST数据集所有像素的均值，0.3081为标准差
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载MNIST训练集
# root：数据存放路径
# train=True：加载训练集
# download=True：如果本地没有数据则自动下载
# transform：应用上面定义的预处理
train_dataset = datasets.MNIST(
    root='D:/PythonCode/Pytorch_learning/MNIST',
    train=True,
    download=True,
    transform=transform
)

# 构建训练集的数据加载器
# DataLoader会自动将数据分成batch，并支持多线程加载和数据打乱
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True  # 每个epoch前打乱数据，提升泛化能力
)

# 加载MNIST测试集
test_dataset = datasets.MNIST(
    root='D:/PythonCode/Pytorch_learning/MNIST',
    train=False,
    download=True,
    transform=transform
)

# 构建测试集的数据加载器
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False  # 测试集不打乱，保证评估一致性
)

# =========================
# 2. 定义神经网络结构
# =========================

class Net(torch.nn.Module):
    """
    定义一个多层全连接神经网络用于MNIST手写数字分类。
    输入层：28x28=784个特征
    隐藏层1：512个神经元
    隐藏层2：256个神经元
    隐藏层3：128个神经元
    隐藏层4：64个神经元
    输出层：10个神经元（对应数字0-9的类别）
    """
    def __init__(self):
        super(Net, self).__init__()  # 初始化父类
        # 定义第1个全连接层，输入784维，输出512维
        self.l1 = torch.nn.Linear(784, 512)
        # 定义第2个全连接层，输入512维，输出256维
        self.l2 = torch.nn.Linear(512, 256)
        # 定义第3个全连接层，输入256维，输出128维
        self.l3 = torch.nn.Linear(256, 128)
        # 定义第4个全连接层，输入128维，输出64维
        self.l4 = torch.nn.Linear(128, 64)
        # 定义第5个全连接层，输入64维，输出10维（10个类别）
        self.l5 = torch.nn.Linear(64, 10)

    def forward(self, x):
        """
        前向传播过程
        :param x: 输入张量，形状为(batch_size, 1, 28, 28)
        :return: 输出张量，形状为(batch_size, 10)
        """
        # 将输入的图片展平成一维向量（batch_size, 784）
        x = x.view(-1, 784)
        # 第1层全连接 + ReLU激活
        x = F.relu(self.l1(x))
        # 第2层全连接 + ReLU激活
        x = F.relu(self.l2(x))
        # 第3层全连接 + ReLU激活
        x = F.relu(self.l3(x))
        # 第4层全连接 + ReLU激活
        x = F.relu(self.l4(x))
        # 第5层全连接（输出层），不加激活，输出原始logits
        x = self.l5(x)
        return x

# 实例化神经网络模型
model = Net()

# =========================
# 3. 定义损失函数和优化器
# =========================

# 交叉熵损失函数，适用于多分类任务
# 输入为网络输出的logits和真实标签
criterion = torch.nn.CrossEntropyLoss()

# 随机梯度下降（SGD）优化器
# model.parameters()：需要优化的参数
# lr=0.01：学习率
# momentum=0.5：动量因子，加速收敛
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# =========================
# 4. 训练函数
# =========================

def train(epoch):
    """
    训练模型一个epoch
    :param epoch: 当前训练轮数（从0开始）
    """
    running_loss = 0.0  # 累计损失，用于统计和打印
    # enumerate(train_loader, 0)：遍历训练集的每个batch，batch_idx为批次索引
    for batch_idx, data in enumerate(train_loader, 0):
        # data是一个元组(inputs, target)
        # inputs: 形状(batch_size, 1, 28, 28)
        # target: 形状(batch_size,)
        inputs, target = data
        optimizer.zero_grad()  # 梯度清零，防止梯度累加
        outputs = model(inputs)  # 前向传播，得到预测结果logits
        loss = criterion(outputs, target)  # 计算当前batch的损失
        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 更新模型参数

        running_loss += loss.item()  # 累加当前batch的损失值
        # 每300个batch打印一次平均损失
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0  # 重置累计损失

# =========================
# 5. 测试函数
# =========================

def test():
    """
    在测试集上评估模型的准确率
    """
    correct = 0  # 预测正确的样本数
    total = 0    # 总样本数
    # torch.no_grad()：测试时不计算梯度，节省内存和加快速度
    with torch.no_grad():
        for data in test_loader:
            images, labels = data  # images: 测试图片, labels: 真实标签
            outputs = model(images)  # 前向传播，得到logits
            # torch.max(outputs.data, dim=1)：返回每行最大值和对应的索引
            # predicted为预测的类别（0-9）
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)  # 累加总样本数
            # (predicted == labels)为布尔张量，sum()统计预测正确的数量
            correct += (predicted == labels).sum().item()
    # 打印测试集上的准确率
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

# =========================
# 6. 主程序入口
# =========================

if __name__ == '__main__':
    # 训练和测试10个epoch
    for epoch in range(10):
        train(epoch)  # 训练一个epoch
        test()        # 在测试集上评估模型