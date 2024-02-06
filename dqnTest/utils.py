# 包含用于辅助功能的实用程序函数，例如日志记录、参数处理等
import logging
import argparse


def setup_logger(log_file='training.log'):
    # 设置日志记录
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')


def parse_arguments():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Deep Q Network for Breakout')
    parser.add_argument('--num_episodes', type=int, default=1000, help='Number of episodes for training')
    parser.add_argument('--max_steps', type=int, default=10000, help='Maximum number of steps per episode')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    return parser.parse_args()
