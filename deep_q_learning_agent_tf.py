import cv2
import gym
import random
import time

import numpy as np

from collections import deque
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from tensorflow.keras.optimizers import Adam

MIN_REPLAY_MEMORY_SIZE = 2048
MINIBATCH_SIZE = 64
DISCOUNT = 0.99
UPDATE_TARGET_EVERY = 4
NO_EPISODES = 20


class DQNAgent:
    def __init__(self, replay_memory_size, states_shape, actions_shape, epsilon):
        self.states_shape = states_shape
        self.actions_shape = actions_shape
        self.epsilon = epsilon
        self.model = self.create_model()
        # Target network

        self.replay_memory = deque(maxlen=replay_memory_size)

        # Used to count when to update target network with main network's weights
        self.train_counter = 0

    def create_model(self):
        model = Sequential()

        model.add(Conv2D(32, (3, 3), strides=(2, 2),
                         input_shape=self.states_shape))  # (80, 80, 1)
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.2))

        model.add(Conv2D(16, (3, 3), strides=(2, 2)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.2))

        model.add(Flatten())  # flattens 3D feature maps to 1D feature vectors
        # model.add(Dense(32))

        model.add(Dense(self.actions_shape, activation='linear'))  # ACTION_SPACE_SIZE = how many choices (7)
        model.compile(loss="mse", optimizer=Adam(lr=0.0005), metrics=['accuracy'])
        model.summary()
        return model

    def update_replay_memory(self, previous_state, action, reward, current_state, done):
        self.replay_memory.append([previous_state, action, reward, current_state, done])

    def get_action(self, state):
        self.epsilon -= 0.0001
        if random.random() < self.epsilon:
            return random.randint(0, self.actions_shape - 1)
        return np.argmax(self.get_q_output(state))

    def get_q_output(self, state):
        state = np.reshape(state, (1, *state.shape))
        prediction = self.model.predict(state, batch_size=1) / 255
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
        previous_states = np.array([mem_item[0] for mem_item in minibatch]) / 255
        previous_q_values = self.model.predict(previous_states)

        # When using target network, query it, otherwise main network should be queried
        current_states = np.array([mem_item[3] for mem_item in minibatch]) / 255
        current_q_values = self.model.predict(current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (previous_state, action, reward, current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_current_q = np.max(current_q_values[index])
                new_q = reward + DISCOUNT * max_current_q
            else:
                new_q = reward

            # Update Q value for given state
            previous_qs = previous_q_values[index]
            previous_qs[action] = new_q

            # And append to our training data
            X.append(previous_state)
            y.append(previous_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X) / 255, np.array(y), epochs=1, batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False)


def observation_to_state(observation, previous_observation):
    # Simplify observation by reducing dimensions and turn in to a 80x80x1 state
    state = np.stack([observation, previous_observation], axis=3)
    state = state[75:235, :, :, :]
    state = np.sum(state, 2) / 3  # grayscale
    state = cv2.resize(state, dsize=(80, 80), interpolation=cv2.INTER_LINEAR)
    return state


def main():
    env = gym.make('Assault-v0')
    agent = DQNAgent(replay_memory_size=2048, states_shape=(80, 80, 2), actions_shape=6, epsilon=0.2)
    for episode in range(NO_EPISODES):
        observation = env.reset()
        current_state = observation_to_state(observation, observation)
        time_step, previous_lives = 0, 4
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
                reward = -5
            agent.update_replay_memory(previous_state, action, reward, current_state, done)
            print(episode, time_step)
            agent.train()

            if done:
                print("Episode finished after {} time steps".format(time_step))
                break
    env.close()


if __name__ == "__main__":
    main()
