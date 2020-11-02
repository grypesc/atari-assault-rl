import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import safe_mean


class StopTrainingOnMaxEpisodes(BaseCallback):

    def __init__(self, env, max_episodes, map_score=lambda x: x, verbose=0):
        super(StopTrainingOnMaxEpisodes, self).__init__(verbose=verbose)
        self.max_episodes = max_episodes
        self._total_max_episodes = max_episodes
        self.n_episodes = 0
        self.env = env
        self.map_score = map_score

    def _init_callback(self) -> None:
        self._total_max_episodes = self.max_episodes * self.training_env.num_envs

    def _on_step(self) -> bool:
        done_array = np.array(self.locals.get("done") if self.locals.get("done") is not None else self.locals.get("dones"))
        episode_change = np.sum(done_array).item()
        if episode_change > 0:

            # DQN-style-reward
            reward = safe_mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
            # DQN-style-length
            length = safe_mean([ep_info["l"] for ep_info in self.model.ep_info_buffer])
            episode_rewards, episode_lengths = evaluate_policy(self.model, self.env, n_eval_episodes=5)
            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            print("episode ended with score {} and length {}".format(reward, length))

        self.n_episodes += episode_change
        continue_training = self.n_episodes < self._total_max_episodes

        if not continue_training:
            print("learning stopped, maximum no. of episodes reached")
        return continue_training
