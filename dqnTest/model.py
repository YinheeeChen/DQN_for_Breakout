# 定义了QNetwork类，它是一个简单的卷积接收状态并输出每个动作的q值的神经网络。
# QNetwork类继承自nn.Module类，它包含一个前向方法，该方法接收状态并返回每个动作的q值。

import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    def __init__(self, obs_shape, num_actions, lr=1e-4):
        super(QNetwork, self).__init__()
        self.obs_shape = obs_shape
        self.num_actions = num_actions
        self.conv_net = torch.nn.Sequential(
            torch.nn.Conv2d(4, 16, (8, 8), stride=(4, 4)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, (4, 4), stride=(2, 2)),
            torch.nn.ReLU(),
        )
# 通过调用conv_net的forward方法，我们可以得到一个张量，它的形状是(batch_size, 32, 7, 7)。
        with torch.no_grad():
            dummy = torch.zeros((1, *obs_shape))
            x = self.conv_net(dummy)
            s = x.shape
            fc_size = s[1] * s[2] * s[3]

# 接下来，我们定义了一个全连接网络，它接收卷积网络的输出，并输出每个动作的q值。
        self.fc_net = torch.nn.Sequential(
            torch.nn.Linear(fc_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, num_actions),
        )
        self.output_size = num_actions
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

# 前向方法接收一个张量x，它的形状是(batch_size, 4, 84, 84)。
    def forward(self, x):
        x /= 255
        conv_latent = self.conv_net(x)
        return self.fc_net(conv_latent.view((conv_latent.shape[0], -1)))
