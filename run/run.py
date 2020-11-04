import time
import os

from agent.baselines3_a2c_agent import A2CAgent
from agent.baselines3_dqn_agent import DQNAgent
from agent.baselines3_ppo_agent import PPOAgent
from agent.baselines3_sac_agent import SACAgent
from agent.random_agent import RandomAgent
from agent.deep_q_learning_agent_torch import run_trained as run_custom_trained
from settings import MODELS_ROOT


def episode_callback(score, env_info, time_steps):
    print("Episode finished with score of {} after {} time steps".format(score, time_steps))


def time_step_callback(rewards, env_info, time_steps):
    pass


def run_trained(agent, sleep=0.05, episodes=1):
    env = agent.env
    for e in range(episodes):
        obs = env.reset()
        while True:
            action = agent.predict_action(obs)
            obs, reward, done, info = env.step(action)
            env.render()
            time.sleep(sleep)
    env.close()


run_trained(RandomAgent())
run_trained(DQNAgent())
run_trained(A2CAgent())
run_trained(PPOAgent())
run_custom_trained(model_path=os.path.join(MODELS_ROOT, "custom_ep_63.pth"))
run_custom_trained(model_path=os.path.join(MODELS_ROOT, "custom_ep_89_forgetting.pth"))
