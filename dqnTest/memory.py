# 实现经验回放缓冲区，用于存储先前的观察和动作，以便进行训练时使用。
import random
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


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.memory = []  # 存储先前的观察和动作
        self.position = 0  # 记录当前的位置

    # 将一个新的观察和动作添加到缓冲区中。
    def add(self, state, action, reward, next_state, done):

        transition = (state, action, reward, next_state, done)

        # 更新位置
        self.position = (self.position + 1) % self.capacity
        # 如果缓冲区未满，则将新的观察和动作添加到列表memory中；
        # 否则，将新的观察和动作替换掉列表memory中的一个元素。
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        # 否则，将新的观察和动作替换掉列表memory中的一个元素。
        else:
            self.memory[self.position] = transition

    # 在您的代码中的 sample 方法中使用 pad_to_fixed_shape
    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

        # 打印 state_batch 中的每个元素
        for i, state in enumerate(state_batch):
            print(f"Element {i + 1} in state_batch: {state.shape if state is not None else None}")
        state_batch = [state for state in state_batch if state is not None]  # 过滤掉 state_batch 中的所有 None 元素

        # # 找到 state_batch 中最大的形状
        # max_state_shape = np.max([state.shape for state in state_batch if state is not None], axis=0)
        #
        # # 将 state_batch 中的所有数组填充为最大形状
        # state_batch_padded = [self.pad_to_fixed_shape(state, max_state_shape) for state in state_batch if state is not None]

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
