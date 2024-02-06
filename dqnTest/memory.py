# 实现经验回放缓冲区，用于存储先前的观察和动作，以便进行训练时使用。
import random
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.memory = []  # 存储先前的观察和动作
        self.position = 0  # 记录当前的位置

    # 将一个新的观察和动作添加到缓冲区中。
    def add(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        # 如果缓冲区未满，则将新的观察和动作添加到列表memory中；
        # 否则，将新的观察和动作替换掉列表memory中的一个元素。
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        # 否则，将新的观察和动作替换掉列表memory中的一个元素。
        else:
            self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    # 从缓冲区中抽取一个随机的批次。
    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

        return (
            np.stack(state_batch),
            np.array(action_batch),
            np.array(reward_batch),
            np.stack(next_state_batch),
            np.array(done_batch)
        )

    # __len__：返回缓冲区的当前大小。
    def __len__(self):
        return len(self.memory)
