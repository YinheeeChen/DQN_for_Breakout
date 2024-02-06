# 用于对游戏帧进行预处理的文件。执行图像处理、降采样。
from PIL import Image
import numpy as np


def preprocess_frame(frame):
    # 将图像转换为灰度图
    frame = Image.fromarray(frame).convert('L')

    # 调整图像大小
    frame = frame.resize((84, 110), Image.ANTIALIAS)

    # 裁剪图像
    frame = frame.crop((0, 26, 84, 110))

    # 将图像转换为 NumPy 数组
    frame = np.array(frame)

    # 二值化图像
    frame[frame < 128] = 0
    frame[frame >= 128] = 255

    # 缩放图像
    frame = frame / 255.0

    return frame
