import time
from settings import MODELS_ROOT
from util.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import DQN


MODEL_PATH = '{}/baselines3_dqn_agent'.format(MODELS_ROOT)


def create_env():
    return VecFrameStack(make_atari_env('Assault-v0', n_envs=1, seed=0), n_stack=4)


def train(time_steps, save=False, **params):
    verbose = params.get('verbose', 1)
    buffer_size = params.get('buffer_size', 10000)
    learning_starts = params.get('learning_starts', 1024)

    # There already exists an environment generator
    # that will make and wrap atari environments correctly.
    # Here we are also multi-worker training (n_envs=4 => 4 environments)
    # Frame-stacking with 4 frames

    env = create_env()
    model = DQN('CnnPolicy', env, verbose=verbose, buffer_size=buffer_size, learning_starts=learning_starts)
    model.learn(time_steps)
    if save:
        model.save(MODEL_PATH)


def run_trained(episode_callback, time_step_callback, sleep=0.1, episodes=100):
    env = create_env()
    model = DQN.load(MODEL_PATH)
    for e in range(episodes):
        obs = env.reset()
        time_step = 0
        while True:
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            env.render()
            time.sleep(sleep)
            time_step += 1
            time_step_callback(rewards, info, time_step)
            if done:
                episode_callback(rewards, info, time_step)
                break
    env.close()