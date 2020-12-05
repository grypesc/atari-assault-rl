# Reinforcement Learning - Atari Assault

## Requirements
1. Python `>= 3.6`
1. PIP `>= 20.2.4`

## Installing dependencies
`pip install -r requirements.txt`

## Running
1. Make sure you have copied all models to `{project root}/models` directory ([link](https://drive.google.com/file/d/1AL8IWu6C206qwLqc1nRVeNs6vt-BsRMU/view?usp=sharing))
1. Decide which agent you want to run. Available agents:
    * `rand` - Random Agent
    * `dqn` - DQN Agent (Baselines3)
    * `a2c` - A2C Agent (Baselines3)
    * `ppo` - PPO Agent
    * `dqn_custom` - DQN Agent (implemented with PyTorch)
    * `dqn_forgetting` - DQN Agent (beta version; with open issues)
1. Decide how long the algorithm should sleep between time steps (default value is 0.05s)
1. Decide how many episodes (full games) the algorithm should play (default value is 1)
1. Run with `python run.py --agent <chosen agent> [--sleep <float>] [--episodes <int>]`
