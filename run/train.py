from agent import baselines3_dqn_agent

baselines3_dqn_agent.train(100, verbose=1, buffer_size=10000, learning_starts=1024)
