# 包含与 Gymnasium 环境交互的代码，负责与 Breakout 环境进行通信。
import gymnasium as gym


class Environment:
    def __init__(self, game_name):
        self.env = gym.make(game_name, render_mode='human', full_action_space=False,
                            repeat_action_probability=0.1,
                            obs_type='rgb')
        self.action_space_size = self.env.action_space.n
        self.observation_space_shape = self.env.observation_space.shape

    # 重置环境
    def reset(self):
        self.env.reset()

    # 执行一个动作
    def step(self, action):
        return self.env.step(action)

    # 返回环境的观察
    def render(self):
        self.env.render()

    # 关闭环境
    def close(self):
        self.env.close()
