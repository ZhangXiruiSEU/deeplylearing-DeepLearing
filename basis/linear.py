import torch
import torch.nn as nn  # 神经网络类
import torch.optim as optim  # 神经网络类

#数据设成向量的形式，用矩阵运算代替for循环
#batch大，计算速度快
#batch小，收敛速度快
x_data=torch.Tensor([[1],[2],[3]])
y_data=torch.Tensor([[5],[10],[15]])


class Model(nn.Module):
    # 构造函数
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        # 前两个参数 输入、输出样本的维度数，第三个bias=True

    # 前馈
    # 后馈由pytorch根据已构建计算图，自动实现；或者使用继承functions的类构建计算块，人工设计反向传播
    def forward(self, x):
        y_pred = self.linear(x)  # 魔术方法
        return y_pred
        """
        魔术方法
         def  __call__(self,*args,**kwargs)  传参： args元组普通参数  kwargs 字典  关键参数
            
            
        
        """

        r"""Base class for all neural network modules.

        Your models should also subclass this class.

        Modules can also contain other Modules, allowing to nest them in
        a tree structure. You can assign the submodules as regular attributes::

            import torch.nn as nn
            import torch.nn.functional as F

            class Model(nn.Module):
                def __init__(self):
                    super(Model, self).__init__()
                    self.conv1 = nn.Conv2d(1, 20, 5)
                    self.conv2 = nn.Conv2d(20, 20, 5)

                def forward(self, x):
                    x = F.relu(self.conv1(x))
                    return F.relu(self.conv2(x))

        Submodules assigned in this way will be registered, and will have their
        parameters converted too when you call :meth:`to`, etc.

        :ivar training: Boolean represents whether this module is in training or
                        evaluation mode.
        :vartype training: bool
        """


model = Model()
criterion = nn.MSELoss(size_average=False)
optimizer = optim.Adagrad(model.parameters(), lr=3)

for epoch in range(100):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(model.linear.weight.item())