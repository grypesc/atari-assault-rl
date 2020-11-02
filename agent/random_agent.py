import gym
import time


def run_trained(episode_callback, time_step_callback, sleep=0.1, episodes=100):
    env = gym.make('Assault-v0')
    for e in range(episodes):
        env.reset()
        time_step = 0
        while True:
            action = env.action_space.sample()
            obs, rewards, done, info = env.step(action)
            env.render()
            time.sleep(sleep)
            time_step += 1
            time_step_callback(rewards, info, time_step)
            if done:
                episode_callback(rewards, info, time_step)
                break
    env.close()

