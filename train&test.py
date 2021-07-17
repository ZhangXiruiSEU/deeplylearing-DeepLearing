import torch
import torch.nn as nn  # 神经网络类
import torch.nn.functional as F  # 激活函数等等
import torch.optim as optim  # 优化器
import torch.utils.data
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms  # 图像处理工具
import resnet
import torchvision
train_dataset = datasets.CIFAR100(root='./dataset/CIFAR100',
                                  train=True,
                                  transform=transforms.ToTensor(),
                                  download=True)
test_dataset = datasets.CIFAR100(root='./dataset/CIFAR100',
                                 train=False,
                                 transform=transforms.ToTensor(),
                                 download=True)

batch_size = 256

resnet152 = resnet.ResNet(block=resnet.Bottleneck, layers=[3, 8, 36, 3],num_class=100)
#model = resnet152
model = torchvision.models.resnet152()
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device=device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), weight_decay=0.0001,momentum=0.9)


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


if __name__ == '__main__':
    for epoch in range(20):
        train(epoch)
        test()
        state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        # torch.save(state, './weight/weight.pth')
