from agent.baselines3_dqn_agent import DQNAgent
from agent.baselines3_a2c_agent import A2CAgent
from agent.random_agent import RandomAgent


A2CAgent.train(100, save=True, verbose=1)
DQNAgent.train(100, save=True, verbose=1, buffer_size=10000, learning_starts=1024)
RandomAgent.train(100)
