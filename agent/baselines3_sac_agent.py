from settings import MODELS_ROOT, TB_LOGS_ROOT
from stable_baselines3 import SAC
from stable_baselines3.sac import CnnPolicy
from util.env_util import make_atari_env

MODEL_PATH = '{}/baselines3_sac_agent'.format(MODELS_ROOT)
TB_LOGS = '{}/baselines3'.format(TB_LOGS_ROOT)


class SACAgent:
    # FIXME Looks like it cannot be used in our case, doesn't support discrete actions

    @staticmethod
    def create_env():
        return make_atari_env('Assault-v0', n_envs=1, seed=0)

    @staticmethod
    def train(time_steps, save=False, **params):
        env = SACAgent.create_env()
        model = SAC(CnnPolicy, env, verbose=params.get('verbose', 1), tensorboard_log=TB_LOGS)
        model.learn(total_timesteps=time_steps)
        if save:
            model.save(MODEL_PATH)

    def __init__(self):
        self.env = SACAgent.create_env()
        self.model = SAC.load(MODEL_PATH)

    def predict_action(self, obs):
        return self.model.predict(obs)[0]

    def map_reward(self, reward):
        return reward.sum()
