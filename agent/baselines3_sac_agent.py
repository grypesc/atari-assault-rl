from settings import MODELS_ROOT
from stable_baselines3 import SAC
from stable_baselines3.sac import MlpPolicy

from settings import MODELS_ROOT
from util.env_util import make_atari_env

MODEL_PATH = '{}/baselines3_sac_agent'.format(MODELS_ROOT)


# FIXME: MemoryError: Unable to allocate 6.57 GiB for an array with shape (1000000, 1, 1, 84, 84) and data type uint8
class SACAgent:

    @staticmethod
    def create_env():
        return make_atari_env('Assault-v0', n_envs=1, seed=0)

    @staticmethod
    def train(time_steps, save=False, **params):
        env = SACAgent.create_env()
        model = SAC(MlpPolicy, env, verbose=params.get('verbose', 1))
        model.learn(total_timesteps=time_steps, log_interval=4)
        if save:
            model.save(MODEL_PATH)

    def __init__(self):
        self.env = SACAgent.create_env()
        self.model = SAC.load(MODEL_PATH)

    def predict_action(self, obs):
        return self.model.predict(obs)[0]

    def map_reward(self, reward):
        return reward.sum()
