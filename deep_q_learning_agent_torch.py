import cv2
import gym
import random
import time

import numpy as np

from collections import deque
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

MIN_REPLAY_MEMORY_SIZE = 10000
MINIBATCH_SIZE = 32
DISCOUNT = 0.99
NO_EPISODES = 50


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=(3, 3))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=(3, 3))
        self.fc1 = nn.Linear(5184, 100)
        self.fc2 = nn.Linear(100, 6)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 5184)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DQNAgent:
    def __init__(self, replay_memory_size, epsilon):
        self.epsilon = epsilon
        self.model = Net()
        # Target network

        self.replay_memory = deque(maxlen=replay_memory_size)
        self.train_counter = 0

    def update_replay_memory(self, previous_state, action, reward, current_state, done):
        self.replay_memory.append([previous_state, action, reward, current_state, done])

    def get_action(self, state):
        self.epsilon -= 1/50_000
        self.epsilon = max(0.05, self.epsilon)
        if random.random() < self.epsilon:
            return random.randint(0, 5)
        _, index = torch.max(self.get_q_output(state), 0)
        return index

    def get_q_output(self, state):
        with torch.no_grad():
            prediction = self.model(state / 255)
        return prediction[0]

    def train(self):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        self.train_counter += 1
        if self.train_counter % 4 != 0:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get previous states from minibatch, then calculate Q values
        previous_states = torch.Tensor(MINIBATCH_SIZE, 2, 80, 80)
        torch.cat([mem_item[0] for mem_item in minibatch], dim=0, out=previous_states)

        previous_states /= 255
        with torch.no_grad():
            previous_q_values = self.model(previous_states)

        current_states = torch.Tensor(MINIBATCH_SIZE, 2, 80, 80)
        torch.cat([mem_item[0] for mem_item in minibatch], dim=0, out=current_states)
        current_states /= 255
        with torch.no_grad():
            current_q_values = self.model(current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (previous_state, action, reward, current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_current_q, _ = torch.max(current_q_values[index], 0)
                new_q = reward + DISCOUNT * max_current_q
            else:
                new_q = reward

            # Update Q value for given state
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


def observation_to_state(observation, previous_observation):
    # Simplify observation by reducing dimensions and turn in to a 80x80x1 state
    state = np.stack([observation, previous_observation], axis=0)
    state = state[:, 75:235, :, :]
    state = np.sum(state, 3) / 3  # grayscale
    state1 = cv2.resize(state[0, :, :], dsize=(80, 80), interpolation=cv2.INTER_LINEAR)
    state2 = cv2.resize(state[1, :, :], dsize=(80, 80), interpolation=cv2.INTER_LINEAR)
    state = np.stack([state1, state2], axis=0)
    state = np.reshape(state, (1, *state.shape))
    return torch.from_numpy(state).float()


def main():
    env = gym.make('Assault-v0')
    agent = DQNAgent(replay_memory_size=10000,  epsilon=0.5)
    time_step = 0
    for episode in range(NO_EPISODES):
        observation = env.reset()
        current_state = observation_to_state(observation, observation)
        previous_lives = 4
        while True:
            env.render()
            action = agent.get_action(current_state)
            previous_state = current_state
            previous_observation = observation
            current_observation, reward, done, info = env.step(action + 1)
            current_state = observation_to_state(current_observation, previous_observation)
            time_step += 1
            reward /= 21
            if previous_lives > info["ale.lives"]:
                previous_lives = info["ale.lives"]
                reward = -100
            agent.update_replay_memory(previous_state, action, reward, current_state, done)
            agent.train()
            print(episode, time_step)

            if done:
                print("Episode finished")
                break
        torch.save(agent.model.state_dict(), "lol.pth")
    env.close()


if __name__ == "__main__":
    main()
