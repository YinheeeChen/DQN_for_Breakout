# DQN for Breakout

## Introduction
This project implements a Deep Q-Network (DQN) to play and train the game Breakout. The implementation includes both CPU and GPU versions for training.

## Features
- **AI Gameplay**: Implements AI to play Breakout.
- **CPU Training**: Provides a CPU version for training the DQN on Breakout.
- **GPU Training**: Provides a GPU version for training the DQN on Breakout.

## Usage
To use the DQN for Breakout, follow these instructions:

1. Train the model on CPU:
   ```sh
   python train_cpu.py
   ```

2. Train the model on GPU:
   ```sh
   python train_gpu.py
   ```

3. Use the trained model to play Breakout:
   ```sh
   python play.py
   ```

## Code Structure
- **train_cpu.py**: Script to train the DQN model on a CPU.
- **train_gpu.py**: Script to train the DQN model on a GPU.
- **play.py**: Script to use the trained model to play Breakout.
- **requirements.txt**: Lists the Python packages required for the project.
