from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack

from settings import MODELS_ROOT, TB_LOGS_ROOT
from util.env_util import make_atari_env

MODEL_PATH = '{}/baselines3_ppo_agent'.format(MODELS_ROOT)
TB_LOGS = '{}/baselines3'.format(TB_LOGS_ROOT)


class PPOAgent:

    @staticmethod
    def create_env(n=1):
        return VecFrameStack(make_atari_env('Assault-v0', n_envs=n, seed=0), n_stack=4)

    @staticmethod
    def train(time_steps, save=False, **params):
        env = PPOAgent.create_env(1)
        model = PPO('CnnPolicy', env, verbose=params.get('verbose', 1), tensorboard_log=TB_LOGS)
        model.learn(total_timesteps=time_steps)
        if save:
            model.save(MODEL_PATH)

    def __init__(self):
        self.env = PPOAgent.create_env(1)
        self.model = PPO.load(MODEL_PATH)

    def predict_action(self, obs):
        return self.model.predict(obs)[0]
