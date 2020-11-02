from settings import MODELS_ROOT, TB_LOGS_ROOT
from stable_baselines3.ppo import CnnPolicy, PPO
from util.env_util import make_vec_env

MODEL_PATH = '{}/baselines3_ppo_agent'.format(MODELS_ROOT)
TB_LOGS = '{}/baselines3'.format(TB_LOGS_ROOT)


# FIXME: MemoryError: Unable to allocate 3.66 GiB for an array with shape (2048, 4, 3, 250, 160) and data type float32
class PPOAgent:

    @staticmethod
    def create_env(n=1):
        return make_vec_env('Assault-v0', n_envs=n)

    @staticmethod
    def train(time_steps, save=False, **params):
        env = PPOAgent.create_env(4)
        model = PPO(CnnPolicy, env, verbose=params.get('verbose', 1), tensorboard_log=TB_LOGS)
        model.learn(total_timesteps=time_steps)
        if save:
            model.save(MODEL_PATH)

    def __init__(self):
        self.env = PPOAgent.create_env()
        self.model = PPO.load(MODEL_PATH)

    def predict_action(self, obs):
        return self.model.predict(obs)[0]

    def map_reward(self, reward):
        return reward.sum()
