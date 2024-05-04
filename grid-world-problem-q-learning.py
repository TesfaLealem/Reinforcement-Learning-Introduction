import gym
import numpy as np

# Hyperparameters
learning_rate = 0.8
gamma = 0.9
epsilon = 0.1  # Exploration rate
max_episodes = 1000

env = gym.make('FrozenLake-v1')
n_states = env.observation_space.n
n_actions = env.action_space.n

# Initialize Q-table with zeros (ensure correct dimensions)
Q = np.zeros((n_states, n_actions))

for episode in range(max_episodes):
  # Extract the integer state from the observation (tuple)
  state, _ = env.reset()

  done = False

  while not done:
    # Epsilon-greedy action selection
    if np.random.random() < epsilon:
      action = env.action_space.sample()
    else:
        # Exploit: Choose action with highest Q-value for current state
      action = np.argmax(Q[state, :])

    # Take action and observe outcome
#    next_state, reward, done, _ = env.step(action)
    next_state, reward, done, *extra_info = env.step(action)

    # Update Q-value based on Bellman equation
    max_q_next = np.max(Q[next_state, :])
    Q[state, action] += learning_rate * (reward + gamma * max_q_next - Q[state, action])

    state = next_state  # Update state for next iteration

  # Print episode reward for monitoring progress
  print(f"Episode: {episode+1}, Reward: {reward}")

env.close()
