# 实现经验回放缓冲区，用于存储先前的观察和动作，以便进行训练时使用。
from random import sample
from collections import deque
import numpy as np


# def pad_to_fixed_shape(array, fixed_shape, pad_value=0):
#     """
#     将数组填充为固定形状。
#
#     Parameters:
#     - array: 要填充的数组。
#     - fixed_shape: 固定的目标形状。
#     - pad_value: 用于填充的值。
#
#     Returns:
#     - 填充后的数组。
#     """
#     pad_widths = [(0, max(0, target_size - current_size)) for target_size, current_size in
#                   zip(fixed_shape, array.shape)]
#     padded_array = np.pad(array, pad_width=pad_widths, mode='constant', constant_values=pad_value)
#     return padded_array

# 经验回放缓冲区
# 用于存储先前的观察和动作，以便进行训练时使用。
# 该类的实例将存储先前的观察、动作、奖励、下一个观察和终止标志，并提供方法用于插入和采样经验。
class ReplayBuffer:
    def __init__(self, buffer_size=100000):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        self.idx = 0

    # 插入经验
    def insert(self, sars):
        self.buffer.append(sars)

    # 采样经验
    def sample(self, num_samples):
        if num_samples > len(self.buffer):
            return sample(self.buffer, len(self.buffer))
        return sample(self.buffer, num_samples)

    def __len__(self):
        return len(self.buffer)


# 定义经验类
class Memory:
    def __init__(self, state, action, reward, next_state, done):
        self.state = np.array(state, dtype=np.float32)  # 适当初始化并指定类型
        self.action = action  # 适当初始化
        self.reward = float(reward)  # 适当初始化并指定类型
        self.next_state = np.array(next_state, dtype=np.float32)  # 适当初始化并指定类型
        self.done = bool(done)  # 适当初始化并指定类型
