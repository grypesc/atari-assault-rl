from agent import baselines3_dqn_agent

baselines3_dqn_agent.train(100, save=True, verbose=1, buffer_size=10000, learning_starts=1024)
