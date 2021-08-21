import torch
import torch.nn as nn  # 神经网络类
import torch.nn.functional as F  # 激活函数等等
import torch.optim as optim  # 优化器
import torch.utils.data
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms  # 图像处理工具

batch_size = 64
transform = transforms.Compose([transforms.ToTensor(),  # 转化为Tensor
                                transforms.Normalize((0.1037,), (0.3081,))])  # 归一化 参数：均值；标准差   归一后 均值为0，标准差为1
# transform =transforms.ToTensor()


# 数据集的下载，继承自Dataset类的对象
train_dataset = datasets.MNIST(root='./dataset/mnist',
                               train=True,
                               transform=transform,
                               download=True)
test_dataset = datasets.MNIST(root='./dataset/mnist',
                              train=False,
                              transform=transform,
                              download=True)

# 也可以自定义数据集类
# """
# class Mydataset(Dataset):
#    def __init__(self)# 1 读取所有数据到内存   2定义文件名列表……
#        pass
#    def __getitem__(self, index)#索引取数据
#        pass
#    def __len__(self)#得到数量
#        pass

# """

# 数据加载
# 数据设成向量的形式，用矩阵运算代替for循环，加快计算速度
# batch_size大，计算速度快
# batch_size小，收敛速度快
# 平衡:mini-batch


train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# batch_size 每一小组数据的大小   shuffle  数据顺序是否打乱 num_workers 进程数


# 将模型定义为一个类
# 继承自nn.Module
class Model(nn.Module):
    # 构造函数
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 20, kernel_size=5)
        # 输入维度（输入的二维数据的通道数），输出维度（卷积核个数），kernel_size卷积核大小 ，stride卷积核移动的步长 ，padding图像外围加一圈卷积后大小不变
        self.conv2 = torch.nn.Conv2d(20, 20, kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(2)
        self.linear1 = nn.Linear(320, 100)
        # 前两个参数 输入、输出样本的维度数，第三个默认bias=True
        # 线性层 y=xA'+b  x第一个维度是batch_size，第二个维度是特征维度数
        # 当前张量维度 (batch_size1,20,4,4)  320=20*4*4

        self.linear2 = nn.Linear(100, 10)
        # 分10类所以输出为10维，由softmax转换为概率

    # 前馈
    # 后馈由pytorch根据已构建计算图，自动实现；或者使用继承自functions的类人工设计反向传播
    def forward(self, x):
        batch_size1 = x.size(0)
        # 当前维度 (batch_size1,1,28,28)
        # 28=32-5+1
        x = F.relu(self.pooling(self.conv1(x)))
        # 当前未池化的维度 (batch_size1,20,24,24)
        # 池化后 (batch_size1,20,12,12)
        # 12=24/2

        x = F.relu(self.pooling(self.conv2(x)))
        # 当前未池化的维度 (batch_size1,20,8,8)
        # 池化后 (batch_size1,20,4,4)

        x = x.view(batch_size1, -1)
        # 拉直 view把原来的张量中的数值按照顺序填充给指定的张量格式 -1表示自动计算
        # 如这里 -1的位置计算得到 320=20*4*4

        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


model = Model()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device=device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), weight_decay=0.001)


# lr 学习率  weight_decay权重衰减/L2正则化


# 将单独的一轮循环定义为函数
def train(epoch):
    running_loss = 0.0
    # 单纯用于记录每n个mini_batch迭代loss的和，以便输出

    for batch_idx, data in enumerate(train_loader, 0):
        # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标 ,第二个参数为下标起始位置
        # batch_idx 即 全部数据分成的若干个mini_batch的索引编号，data 即 数据

        inputs, labels = data
        # inputs为输出 labels为对应的输出
        inputs, labels = inputs.to(device), labels.to(device)
        # GPU加速

        outputs = model(inputs)
        # 正向传播，通过定义于基类的__call__()魔术方法自动调用forward()函数

        loss = criterion(outputs, labels)
        # 计算loss

        optimizer.zero_grad()
        # PyTorch梯度值自动累加，这里不需要累加则将其清零

        loss.backward()
        # 反向传播

        optimizer.step()
        # 更新参数， Performs a single optimization step (parameter update).

        running_loss += loss.item()  # .item()  取loss的值

        if batch_idx % 300 == 299:
            print('[%d,%5d] loss:%f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            print(running_loss)
            running_loss = 0.0


# 将测试定义为函数
def test():
    correct = 0
    total = 0
    with torch.no_grad():
        # 上下文管理器(context manager)用于规定某个对象的使用范围 进入或者离开该使用范围，会有特殊操作被调用
        # 在这里使本代码块中的Tensor对象不再计算梯度

        for data in test_loader:
            # 同样是以mini_batch为单位取数据，不过这里不需要记录索引值

            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # GPU加速
            outputs = model(inputs)
            # outputs 竖列为每一条数据；横行为每条数据的输出，10个概率值

            _, predicted = torch.max(outputs.data, dim=1)
            # max()第一个参数为tensor，第二个参数为沿着第几个维度取最大值；第一个返回值为最大值，第二个返回值为下标
            # 将每一行概率最大的下标，即预测的类别取出来,类型仍是矩阵，一组mini_batch中所有数据的预测值

            total += labels.size(0)
            # labels.size(0)测试集mini_batch的数据个数 size为元组(n,1)，取第0元素

            correct += (predicted == labels).sum().item()

    print("accuracy:%f %%" % (100 * correct / total))


test_flag = True
if __name__ == '__main__':

    if test_flag:
        # 加载保存的模型
        checkpoint = torch.load('./weight.pth')
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print('加载 epoch {} 成功！'.format(start_epoch))
    for epoch in range(start_epoch,20):
        train(epoch)
        test()
        state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, './weight/'+'minist'+str(epoch)+'weight.pth')

# https://www.jianshu.com/p/1cd6333128a1  模型保存相关
