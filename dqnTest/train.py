# 包含训练循环的代码，负责在环境中运行代理、收集经验并更新神经网络参数。
# 该文件的代码将在后续的实验中被修改和扩展。
import torch
import numpy as np


def train(env, agent, num_episodes=1000, max_steps=10000, batch_size=32):
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            # 选择动作
            action = agent.select_action(state)

            # 执行动作并观察下一个状态和奖励
            next_state, reward, done, _ = env.step(action)

            # 将经验存储到回放缓冲区
            agent.remember(state, action, reward, next_state, done)

            # 更新神经网络参数
            agent.update_q_network(batch_size)

            # 更新当前状态
            state = next_state
            episode_reward += reward

            # 如果达到终止状态，结束该episode
            if done:
                break

        # 打印每个episode的总奖励
        print(f"Episode {episode + 1}, Reward: {episode_reward}")

    # 保存训练后的模型参数
    torch.save(agent.q_network.state_dict(), 'trained_model.pth')
