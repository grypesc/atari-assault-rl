import time
import gym
from settings import MODELS_ROOT
from util.env_util import make_atari_env
from stable_baselines3 import A2C
from stable_baselines3.a2c import MlpPolicy
from util.env_util import make_vec_env


MODEL_PATH = '{}/baselines3_a2c_agent'.format(MODELS_ROOT)


class A2CAgent:

    @staticmethod
    def create_env(n=1):
        return make_vec_env('Assault-v0', n_envs=n)

    @staticmethod
    def train(time_steps, save=False, **params):
        env = A2CAgent.create_env(4)
        model = A2C(MlpPolicy, env, verbose=params.get('verbose', 1))
        model.learn(total_timesteps=time_steps)
        if save:
            model.save(MODEL_PATH)

    def __init__(self):
        self.env = A2CAgent.create_env()
        self.model = A2C.load(MODEL_PATH)

    def predict_action(self, obs):
        return self.model.predict(obs)[0]

    # FIXME reward is either 0.0 or 1.0
    def map_reward(self, reward):
        return reward.sum()
