from settings import MODELS_ROOT, TB_LOGS_ROOT
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecFrameStack
from util.env_util import make_atari_env

MODEL_PATH = '{}/baselines3_a2c_agent'.format(MODELS_ROOT)
TB_LOGS = '{}/baselines3'.format(TB_LOGS_ROOT)


class A2CAgent:

    @staticmethod
    def create_env(n=1):
        return VecFrameStack(make_atari_env('Assault-v0', n_envs=n, seed=0), n_stack=4)

    @staticmethod
    def train(time_steps, save=False, **params):
        env = A2CAgent.create_env(1)
        model = A2C('CnnPolicy', env, verbose=params.get('verbose', 1), tensorboard_log=TB_LOGS)
        model.learn(total_timesteps=time_steps)
        if save:
            model.save(MODEL_PATH)

    def __init__(self):
        self.env = A2CAgent.create_env(1)
        self.model = A2C.load(MODEL_PATH)

    def predict_action(self, obs):
        return self.model.predict(obs)[0]

    # FIXME reward is either 0.0 or 1.0
    def map_reward(self, reward):
        return reward.sum()
