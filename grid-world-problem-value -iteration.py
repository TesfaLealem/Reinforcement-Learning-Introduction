import gym
import numpy as np

# Create the FrozenLake environment (version 1)
env = gym.make('FrozenLake-v1') 

# Define hyperparameters (adjust as needed)
discount_factor = 0.8  # Discount factor for future rewards
episodes = 1000  # Number of training episodes

# Initialize value table for all states (zeros with float data type)
value_table = np.zeros(env.observation_space.n, dtype=float) 

# Initialize policy table for all states (zeros with integer data type)
policy = np.zeros(env.observation_space.n, dtype=int) 

def value_iteration(env, value_table, discount_factor, episodes):
  """
  This function performs Value Iteration for the FrozenLake environment.

  Args:
      env: The FrozenLake environment object.
      value_table: The current value table for all states.
      discount_factor: The discount factor for future rewards.
      episodes: The number of training episodes.

  Returns:
      A tuple containing the updated value table and policy table.
  """
  for _ in range(episodes):
    # Iterate through all states
    for state in range(env.observation_space.n):
      # Find the best action and expected future reward from this state
      best_action = None
      # Negative infinity for initialization
      max_future_reward = -float('inf')  

      for action in range(env.action_space.n):
        expected_reward = 0
        # Iterate through possible next states and rewards
        for next_state, reward, done, info in env.P[state][action]:
          # Expected reward based on transition probabilities and next state value
          if len(env.P[state][action]) > 0:
            expected_reward += env.P[state][action][0][1] * (
                reward + discount_factor * value_table[env.P[state][action][0][1]]
            )
          else:
            # No transitions, keep current value (default behavior)
            expected_reward = value_table[state]

        if expected_reward > max_future_reward:
          max_future_reward = expected_reward
          # Modulo operation for safety
          best_action = action % env.action_space.n 

      # Update policy with best action for this state
      policy[state] = best_action

      # Update value table for the current state with the maximum expected future reward
      value_table[state] = max_future_reward

  # Return the learned value table and policy table
  return value_table, policy


def evaluate_policy(env, policy, episodes):
  """
  This function evaluates the learned policy on the FrozenLake environment.

  Args:
      env: The FrozenLake environment object.
      policy: The learned policy table.
      episodes: The number of evaluation episodes.

  Returns:
      The average reward achieved over the evaluation episodes.
  """
  total_reward = 0
  for _ in range(episodes):
    state = env.reset()
    done = False
    while not done:
      try:
        action = policy[state]  # Attempt to get action from policy
      except IndexError:
        # Handle potential missing policy for a state (use random action)
        action = env.action_space.sample()  # Random action

      # Take action, observe reward and next state, handle potential extra info
      #state, reward, done, info = env.step(action)  # Unpacks only 4 elements
      state, reward, done, *extra_info = env.step(action)


      # Update total reward
      total_reward += reward
# Average reward
# Initialize a random policy for each state
  return total_reward / episodes  
for state in range(env.observation_space.n):
  policy[state] = env.action_space.sample()  # Random action for each state

# After running value_iteration or policy_iteration function
value_table, policy = value_iteration(env, value_table, discount_factor, episodes) 
# Evaluate over 100 episodes
average_reward = evaluate_policy(env, policy, 100)  
print("Average reward using learned policy:", average_reward)

