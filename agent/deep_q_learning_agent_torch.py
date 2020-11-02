import cv2
import gym
import random
import time

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

from collections import deque
from settings import MODELS_ROOT


class _Net(nn.Module):
    def __init__(self):
        super(_Net, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size=(3, 3))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=(3, 3))
        self.fc2 = nn.Linear(32 * 18 * 18, 6)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.parameters(), lr=0.0005, momentum=0.9)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 32 * 18 * 18)
        x = self.fc2(x)
        return x


class DQNAgent:
    def __init__(self, replay_memory_size, epsilon):
        self.epsilon = epsilon
        self.discount = 0.99
        self.min_replay_memory_size = 8000
        self.mini_batch_size = 64
        self.model = _Net()
        self.replay_memory = deque(maxlen=replay_memory_size)
        self.policy_update_counter = 0

    def update_replay_memory(self, previous_state, action, reward, current_state, done):
        self.replay_memory.append([previous_state, action, reward, current_state, done])

    def predict(self, state):
        self.epsilon = max(0.05, self.epsilon)
        if random.random() < self.epsilon:
            return random.randint(0, 5)
        _, index = torch.max(self.get_q_output(state), 0)
        return index

    def get_q_output(self, state):
        with torch.no_grad():
            prediction = self.model(state / 255)
        return prediction[0]

    def update_policy(self):
        self.epsilon -= 1 / 50_000
        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < self.min_replay_memory_size:
            return
        self.policy_update_counter += 1
        if self.policy_update_counter % 16 != 0:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, self.mini_batch_size)

        # Get previous states from minibatch, then calculate Q values
        previous_states = torch.Tensor(self.mini_batch_size, 2, 80, 80)
        torch.cat([mem_item[0] for mem_item in minibatch], dim=0, out=previous_states)

        previous_states /= 255
        with torch.no_grad():
            previous_q_values = self.model(previous_states)
            current_states = torch.Tensor(self.mini_batch_size, 2, 80, 80)
            torch.cat([mem_item[3] for mem_item in minibatch], dim=0, out=current_states)
            current_states /= 255
            current_q_values = self.model(current_states)

        X = []
        y = []

        for index, (previous_state, action, reward, current_state, done) in enumerate(minibatch):
            if not done:
                max_current_q, _ = torch.max(current_q_values[index], 0)
                new_q = reward + self.discount * max_current_q
            else:
                new_q = reward

            previous_qs = previous_q_values[index]
            previous_qs[action] = new_q

            X.append(previous_state)
            y.append(previous_qs.unsqueeze(0))

        X = torch.cat(X, 0) / 255
        y = torch.cat(y, 0)
        self.model.optimizer.zero_grad()
        outputs = self.model(X)
        loss = self.model.criterion(outputs, y)
        loss.backward()
        self.model.optimizer.step()

    def load(self, path):
        self.model.load_state_dict(torch.load(path))


def observation_to_state(observation, previous_observation):
    # Simplify observation by reducing dimensions and turn in to a 80x80x2 state
    state = np.stack([observation, previous_observation], axis=0)
    state = state[:, 75:235, :, :]
    state = np.sum(state, 3) / 3  # grayscale
    state1 = cv2.resize(state[0, :, :], dsize=(80, 80), interpolation=cv2.INTER_LINEAR)
    state2 = cv2.resize(state[1, :, :], dsize=(80, 80), interpolation=cv2.INTER_LINEAR)
    state = np.stack([state1, state2], axis=0)
    state = np.reshape(state, (1, *state.shape))
    return torch.from_numpy(state).float()


def train(no_episodes=10, save=True, buffer_size=10000):
    env = gym.make('Assault-v0')
    agent = DQNAgent(replay_memory_size=buffer_size, epsilon=0.5)
    time_step = 0
    score = [0 for i in range(no_episodes)]
    for episode in range(no_episodes):
        observation = env.reset()
        current_state = observation_to_state(observation, observation)
        previous_lives = 4
        while True:
            action = agent.predict(current_state)
            previous_state = current_state
            previous_observation = observation
            current_observation, reward, done, info = env.step(action + 1)
            current_state = observation_to_state(current_observation, previous_observation)
            time_step += 1
            score[episode] += reward
            reward /= 21
            if previous_lives > info["ale.lives"]:
                previous_lives = info["ale.lives"]
                reward = -1
            agent.update_replay_memory(previous_state, action, reward, current_state, done)
            agent.update_policy()
            if done:
                print("Episode no {} finished after {} timesteps".format(episode, time_step))
                print("Score: {}".format(score[episode]))
                break
        if save:
            torch.save(agent.model.state_dict(), "{}/custom_dqn_ep_{}.pth".format(MODELS_ROOT, episode))


def run_trained(model_path, episodes=5):
    env = gym.make('Assault-v0')
    agent = DQNAgent(replay_memory_size=10000, epsilon=0.0)
    agent.load(model_path)
    score = 0
    for episode in range(episodes):
        observation = env.reset()
        current_state = observation_to_state(observation, observation)
        while True:
            env.render()
            time.sleep(0.03)
            action = agent.predict(current_state)
            current_observation, reward, done, info = env.step(action + 1)
            score += reward
            if done:
                break
    score /= episodes
    print("Average score: " + str(episode))
    return score


def main():
    # Train and test at the same time
    NO_EPISODES = 50
    NO_TESTING_EPISODES = 5
    env = gym.make('Assault-v0')
    agent = DQNAgent(replay_memory_size=10000, epsilon=0.5)
    time_step = 0
    score = [0 for i in range(NO_EPISODES)]
    for episode in range(NO_EPISODES):
        observation = env.reset()
        current_state = observation_to_state(observation, observation)
        previous_lives = 4
        while True:
            action = agent.predict(current_state)
            previous_state = current_state
            previous_observation = observation
            current_observation, reward, done, info = env.step(action + 1)
            current_state = observation_to_state(current_observation, previous_observation)
            time_step += 1
            reward /= 21
            if previous_lives > info["ale.lives"]:
                previous_lives = info["ale.lives"]
                reward = -1
            agent.update_replay_memory(previous_state, action, reward, current_state, done)
            agent.update_policy()

            if done:
                print("Episode no {} finished after {} timesteps".format(episode, time_step))
                break

        for testing_episode in range(NO_TESTING_EPISODES):
            observation = env.reset()
            current_state = observation_to_state(observation, observation)
            while True:
                # env.render()
                action = agent.predict(current_state)
                current_observation, reward, done, info = env.step(action + 1)
                score[episode] += reward
                if done:
                    break
        score[episode] /= NO_TESTING_EPISODES
        print("Score: " + str(score[episode]))

        torch.save(agent.model.state_dict(), "ep_" + str(episode) + ".pth")
    env.close()
    plt.plot(score)
    plt.show()


if __name__ == "__main__":
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
        device = torch.device(dev)
    train(5)
