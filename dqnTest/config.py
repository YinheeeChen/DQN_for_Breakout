# 存储训练过程中的超参数和配置选项。
# 通过修改这些参数，可以改变训练的行为。
# 这些参数是硬编码在代码中的，但是也可以通过命令行参数或者配置文件来指定。
# 为了简化代码，这里直接硬编码在代码中。
class Config:
    def __init__(self):
        # 网络参数
        self.input_size = (4, 84, 84)  # 输入图像的形状
        self.output_size = 4  # 动作数量

        # 训练参数
        self.num_episodes = 1000  # 训练的总episode数
        self.max_steps = 10000  # 每个episode的最大步数
        self.batch_size = 32  # 每次训练从回放缓冲区中抽取的样本数量

        # DQN参数
        self.gamma = 0.99  # 折扣因子，用于计算未来奖励的折现值
        self.epsilon_start = 1.0  # 初始探索率
        self.epsilon_end = 0.01  # 最终探索率
        self.epsilon_decay = 0.995  # 探索率衰减因子

        # 其他配置
        self.log_file = 'training.log'  # 日志记录文件


config = Config()
