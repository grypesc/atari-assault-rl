from agent import random_agent
from agent import baselines3_dqn_agent


def episode_callback(rewards, env_info, time_steps):
    print("Episode finished with score of {} after {} time steps".format(rewards.max(), time_steps))


def time_step_callback(reward, env_info, time_steps):
    pass


# random_agent.run_trained(episode_callback, time_step_callback)
baselines3_dqn_agent.run_trained(episode_callback, time_step_callback, sleep=0.001)
