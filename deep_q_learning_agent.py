import random

import gym
import time

import numpy as np

from collections import deque
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from tensorflow.keras.optimizers import Adam

MIN_REPLAY_MEMORY_SIZE = 1000
MINIBATCH_SIZE = 32
DISCOUNT = 0.99


class DQNAgent:
    def __init__(self, replay_memory_size=10000):
        self.model = self.create_model()

        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=replay_memory_size)

        # Custom tensorboard object
        # self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def create_model(self, states_tensor_shape, actions_tensor_shape):
        model = Sequential()

        model.add(Conv2D(128, (3, 3),
                         input_shape=states_tensor_shape))  # states_space_shape = (10, 10, 3) a 10x10 RGB image.
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(128, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Flatten())  # flattens 3D feature maps to 1D feature vectors
        model.add(Dense(64))

        model.add(Dense(actions_tensor_shape, activation='linear'))  # ACTION_SPACE_SIZE = how many choices (5)
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_q_output(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape) / 255)[0]

    # Trains main network every step during episode
    def train(self, terminal_state, step):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch]) / 255
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch]) / 255
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X) / 255, np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0


def observation_to_state(observation):
    # Simplify observation by reducing dimensions and turn in to a 160x160 state
    state = observation[75:235, :, :]
    state = np.sum(state, 2) / 3  # grayscale
    return state


if __name__ == "__main__":
    env = gym.make('Assault-v0')
    agent = DQNAgent()

    for i_episode in range(20):
        observation = env.reset()
        time_step = 0
        while True:
            env.render()
            time.sleep(0.03)
            action = np.argmax(agent.get_q_output(state))
            observation, reward, done, info = env.step(action)
            state
            time_step += 1

            print(observation.shape, reward, done, info)
            if done:
                print("Episode finished after {} time steps".format(time_step + 1))
                break
    env.close()
