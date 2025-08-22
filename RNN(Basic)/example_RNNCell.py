import torch

input_size = 4
hidden_size = 4
batch_size = 1

idx2char = ['e', 'h', 'l', 'o'] # 字母列表
x_data = [1, 0, 2, 2, 3] # The input sequence is 'hello'
y_data = [3, 1, 2, 3, 2] # The output sequence is 'ohlol'

one_hot_lookup = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
x_one_hot = [one_hot_lookup[x] for x in x_data] # 将输入序列转换为独热向量,维度是seq * input_size

inputs = torch.Tensor(x_one_hot).view(-1, batch_size, input_size) # Reshape the inputs to (seqlen, batchSize, inputSize)
labels = torch.LongTensor(y_data).view(-1, 1) # Reshape the labels to (seqlen, 1)

class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(Model, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnncell = torch.nn.RNNCell(input_size = self.input_size, hidden_size = self.hidden_size)

    def forward(self, input, hidden):
        hidden = self.rnncell(input, hidden) # h_t = cell(x_t, h_t-1)
        return hidden

    def init_hidden(self): #生成默认的初始h0
        return torch.zeros(self.batch_size, self.hidden_size)

net = Model(input_size, hidden_size, batch_size)

criterion = torch.nn.CrossEntropyLoss() # 交叉熵损失函数
optimizer = torch.optim.Adam(net.parameters(), lr = 0.1) # 优化器

for epoch in range(15):
    loss = 0
    optimizer.zero_grad() # 优化器梯度归零
    hidden = net.init_hidden() # 初始化hidden(算h0)
    print('Predicted string: ', end='')
    for input, label in zip(inputs, labels): # inputs: seq_len * batch_size * input_size
        hidden = net(input, hidden) # 核心语句,计算h_t
        loss += criterion(hidden, label) # 没有用item(),因为所有序列loss的和才是最终的loss
        _, idx = torch.max(hidden, dim = 1)
        print(idx2char[idx.item()], end='')
    loss.backward()
    optimizer.step()
    print(', epoch [%d/15] loss=%.4f' % (epoch + 1, loss.item()))



