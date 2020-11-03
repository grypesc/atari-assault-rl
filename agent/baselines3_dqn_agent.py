from settings import MODELS_ROOT, TB_LOGS_ROOT
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import DQN
from util.env_util import make_atari_env

MODEL_PATH = '{}/baselines3_dqn_agent'.format(MODELS_ROOT)
TB_LOGS = '{}/baselines3'.format(TB_LOGS_ROOT)


class DQNAgent:

    @staticmethod
    def create_env(n=1):
        return VecFrameStack(make_atari_env('Assault-v0', n_envs=n, seed=0), n_stack=4)

    @staticmethod
    def train(time_steps, save=False, **params):
        verbose = params.get('verbose', 1)
        buffer_size = params.get('buffer_size', 10000)
        learning_starts = params.get('learning_starts', 1024)
        env = DQNAgent.create_env(1)
        model = DQN('CnnPolicy', env, verbose=verbose, buffer_size=buffer_size, learning_starts=learning_starts,
                    tensorboard_log=TB_LOGS)
        model.learn(time_steps)
        if save:
            model.save(MODEL_PATH)

    def __init__(self):
        self.env = DQNAgent.create_env()
        self.model = DQN.load(MODEL_PATH)

    def predict_action(self, obs):
        return self.model.predict(obs)[0]

    def map_reward(self, reward):
        return reward.sum()
