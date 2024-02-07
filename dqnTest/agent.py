# 包含强化学习代理的实现，负责决策、训练和与环境交互。
# 使用 Deep Q Network (DQN) 代理来实现这些功能。
import torch
import torch.optim as optim
import numpy as np
from model import QNetwork
from memory import ReplayBuffer
from memory import Memory


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

        state_transitions = self.memory.sample(batch_size)

        state_batch = torch.stack([torch.tensor(s.state, dtype=torch.float32) for s in state_transitions])
        reward_batch = torch.stack([torch.tensor([s.reward], dtype=torch.float32) for s in state_transitions])
        done_batch = torch.stack([torch.tensor([0]) if s.done else torch.tensor([1]) for s in state_transitions])
        next_state_batch = torch.stack([torch.tensor(s.next_state, dtype=torch.float32) for s in state_transitions])
        action_batch = torch.tensor([s.action for s in state_transitions], dtype=torch.long)

        q_values = self.q_network(state_batch)
        q_values_for_actions = q_values.gather(1, action_batch.unsqueeze(1))

        next_q_values = self.target_q_network(next_state_batch)
        max_next_q_values = torch.max(next_q_values, 1)[0].unsqueeze(1)
        target_q_values = reward_batch + (1 - done_batch) * self.gamma * max_next_q_values

        loss = torch.nn.functional.smooth_l1_loss(q_values_for_actions, target_q_values)
        self.optimizer.zero_grad()

        loss.backward()
        self.optimizer.step()

        self.update_target_q_network()

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_end)

    # 更新目标Q网络：通过从当前Q网络复制参数来更新目标Q网络
    def update_target_q_network(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())

    # 与环境交互：与环境交互并将经验存储在经验回放缓冲区中。
    def remember(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        self.memory.insert(Memory(state, action, reward, next_state, done))
