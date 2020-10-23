import gym
import time

if __name__ == "__main__":
    env = gym.make('Assault-v0')
    for i_episode in range(20):
        observation = env.reset()
        time_step = 0
        while True:
            env.render()
            time.sleep(0.03)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            time_step += 1
            print(observation.shape, reward, done, info)
            if done:
                print("Episode finished after {} time steps".format(time_step + 1))
                break
    env.close()
