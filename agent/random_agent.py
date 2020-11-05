import gym


class RandomAgent:

    @staticmethod
    def create_env():
        return gym.make('Assault-v0')

    @staticmethod
    def train(time_steps, save=False, **params):
        pass

    def __init__(self):
        self.env = RandomAgent.create_env()

    def predict_action(self, obs):
        return self.env.action_space.sample()
