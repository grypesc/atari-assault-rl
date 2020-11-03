from agent.baselines3_a2c_agent import A2CAgent
from agent.baselines3_dqn_agent import DQNAgent
from agent.baselines3_ppo_agent import PPOAgent
from agent.baselines3_sac_agent import SACAgent
from agent.random_agent import RandomAgent

A2CAgent.train(100_0000, save=True, verbose=1)
DQNAgent.train(100_0000, save=True, verbose=1, buffer_size=10000, learning_starts=1024)
# SACAgent.train(10_000, save=True, verbose=1)
PPOAgent.train(100_000, save=True, verbose=1)
# RandomAgent.train(100)
