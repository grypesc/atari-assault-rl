import argparse
import os
import time

from agent.baselines3_a2c_agent import A2CAgent
from agent.baselines3_dqn_agent import DQNAgent
from agent.baselines3_ppo_agent import PPOAgent
from agent.deep_q_learning_agent_torch import run_trained as run_custom_trained
from agent.random_agent import RandomAgent
from settings import MODELS_ROOT


def episode_callback(score, env_info, time_steps):
    print("Episode finished with score of {} after {} time steps".format(score, time_steps))


def time_step_callback(rewards, env_info, time_steps):
    pass


def run_trained(agent, sleep=0.05, episodes=1):
    env = agent.env
    for e in range(episodes):
        obs = env.reset()
        lives = 3
        while True:
            action = agent.predict_action(obs)
            obs, reward, done, info = env.step(action)
            env.render()
            time.sleep(sleep)
            if done:
                lives -= 1
            if done and lives == 0:
                break
    env.close()


agent_runner = {
    'rand': lambda s, e: run_trained(RandomAgent(), s, e),
    'dqn': lambda s, e: run_trained(DQNAgent(), s, e),
    'a2c': lambda s, e: run_trained(A2CAgent(), s, e),
    'ppo': lambda s, e: run_trained(PPOAgent(), s, e),
    'dqn_tf': lambda s, e: run_custom_trained(model_path=os.path.join(MODELS_ROOT, "custom_ep_63.pth"), sleep=s, episodes=e),
    'dqn_t': lambda s, e: run_custom_trained(model_path=os.path.join(MODELS_ROOT, "custom_ep_89_forgetting.pth"), sleep=s, episodes=e)
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, help='agent chosen to run [rand|dqn|a2c|ppo|dqn_tf|dqn_t]')
    parser.add_argument('--sleep', type=float, nargs='?', default=0.05, help='sleep time between time steps')
    parser.add_argument('--episodes', type=int, nargs='?', default=1, help='no. of episodes to run')
    args = parser.parse_args()
    agent_runner[args.agent.lower()](args.sleep, args.episodes)
