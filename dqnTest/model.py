# 定义了QNetwork类，它是一个简单的卷积接收状态并输出每个动作的q值的神经网络。
# QNetwork类继承自nn.Module类，它包含一个前向方法，该方法接收状态并返回每个动作的q值。

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()

        self.output_size = output_size

        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, output_size)

    # 前向方法接收一个状态张量，并将其传递给三个卷积层和两个全连接层。
    # 前向方法返回每个动作的q值。
    # 在前向方法中，使用ReLU激活函数来激活每个卷积层和全连接层。
    # 在前向方法中，使用view方法将卷积层的输出展平为一维张量。
    def forward(self, state):
        x = torch.relu(self.conv1(state))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))

        x = x.view(x.size(0), -1)

        x = torch.relu(self.fc1(x))
        q_values = self.fc2(x)

        return q_values
