# 包含强化学习代理的实现，负责决策、训练和与环境交互。
# 使用 Deep Q Network (DQN) 代理来实现这些功能。
import torch
import torch.optim as optim
import numpy as np
from model import QNetwork
from memory import ReplayBuffer


class DQNAgent:
    def __init__(self, input_size, output_size, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.q_network = QNetwork(input_size, output_size)
        self.target_q_network = QNetwork(input_size, output_size)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.memory = ReplayBuffer()
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.0001)

    # 选择动作：根据当前状态选择动作。
    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(range(self.q_network.output_size))
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                q_values = self.q_network(state_tensor)
                return torch.argmax(q_values).item()

    # 更新Q网络：使用经验回放训练Q网络。
    def update_q_network(self, batch_size):
        if len(self.memory) < batch_size:
            return

        # 从回放缓冲区中采样一批转换
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(batch_size)

        state_batch = torch.tensor(state_batch, dtype=torch.float32)
        action_batch = torch.tensor(action_batch, dtype=torch.long)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32)
        next_state_batch = torch.tensor(next_state_batch, dtype=torch.float32)
        done_batch = torch.tensor(done_batch, dtype=torch.float32)

        # 计算 q 值
        q_values = self.q_network(state_batch)
        q_values_for_actions = q_values.gather(1, action_batch.unsqueeze(1))

        # 使用目标q网络计算目标q值
        next_q_values = self.target_q_network(next_state_batch)
        max_next_q_values = torch.max(next_q_values, 1)[0].unsqueeze(1)
        target_q_values = reward_batch + (1 - done_batch) * self.gamma * max_next_q_values

        # 计算损失并执行梯度下降步骤
        loss = torch.nn.functional.smooth_l1_loss(q_values_for_actions, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新目标Q-network
        self.update_target_q_network()

        # 更新epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_end)

    # 更新目标Q网络：通过从当前Q网络复制参数来更新目标Q网络
    def update_target_q_network(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())

    # 与环境交互：与环境交互并将经验存储在经验回放缓冲区中。
    def remember(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)