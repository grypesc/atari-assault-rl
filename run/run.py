import time

from agent.baselines3_a2c_agent import A2CAgent
from agent.baselines3_dqn_agent import DQNAgent
from agent.baselines3_ppo_agent import PPOAgent
from agent.baselines3_sac_agent import SACAgent
from agent.random_agent import RandomAgent


def episode_callback(score, env_info, time_steps):
    print("Episode finished with score of {} after {} time steps".format(score, time_steps))


def time_step_callback(rewards, env_info, time_steps):
    pass


def run_trained(agent, sleep=0.001, episodes=100):
    env = agent.env
    for e in range(episodes):
        obs = env.reset()
        time_step = 0
        acc_reward = 0
        while True:
            action = agent.predict_action(obs)
            obs, reward, done, info = env.step(action)
            env.render()
            time.sleep(sleep)
            time_step += 1
            acc_reward += agent.map_reward(reward)
            time_step_callback(reward, info, time_step)
            if done:
                episode_callback(acc_reward, info, time_step)
                break
    env.close()


run_trained(DQNAgent())
run_trained(RandomAgent())
run_trained(A2CAgent())
run_trained(SACAgent())
run_trained(PPOAgent())
