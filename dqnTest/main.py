# 主要的入口文件，用于初始化和运行整个训练流程。
from agent import DQNAgent
from environment import Environment
from train import train

if __name__ == "__main__":
    # 初始化环境和代理
    env = Environment('BreakoutNoFrameskip-v4')
    agent = DQNAgent(input_size=env.observation_space_shape, output_size=env.action_space_size)

# 训练代理
    train(env, agent, num_episodes=1000)