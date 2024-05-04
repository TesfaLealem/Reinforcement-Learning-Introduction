import gym
import numpy as np

def policy_iteration(env, discount_factor=0.9, epsilon=0.01):
    """
    Performs policy iteration algorithm on a FrozenLake-v1 environment.

    Args:
        env: FrozenLake-v1 Gym environment.
        discount_factor: Discount factor for future rewards (0 to 1).
        epsilon: Stopping criteria for policy evaluation (change in value table).

    Returns:
        A tuple containing the optimal policy and the final value table.
    """

    # Find goal state (assuming reward is 1.0 for reaching the goal)
    goal_state = None
    for state in range(env.observation_space.n):
        # Check all possible actions
        for action in range(env.action_space.n): 
            # Check reward for any action
            if env.P[state][action][0][3] == 1.0:  
                goal_state = state
                break
        if goal_state is not None:
            break 

    if goal_state is None:
        raise ValueError("Failed to find goal state in FrozenLake environment.")

    # Initialize random policy
    policy = np.random.choice(env.action_space.n, size=env.observation_space.n)

    # Initialize value table with zeros
    value_table = np.zeros(env.observation_space.n)

    while True:
        # Policy evaluation
        value_table_stable = True
        new_value_table = np.zeros_like(value_table)
        for state in range(env.observation_space.n):
            if state == goal_state:
                continue

            # Check for valid action before accessing reward
            if len(env.P[state]) > 0:
                # Initialize for handling states with no valid actions
                max_prob = 0.0  
                for action in range(env.action_space.n):
                    # Validate action index and access reward only if valid
                    if len(env.P[state][action]) > 0 and env.P[state][action][0][3] == 1.0:
                        prob, next_state, reward, _ = env.P[state][action][0]
                        max_prob = max(max_prob, prob)
                        new_value_table[state] = new_value_table[state] + prob * (
                                    reward + discount_factor * value_table[next_state])
                    else:
                        # Handle states with missing actions
                        pass
            else:
                # Handle states with no valid actions
                pass

            # Check for stopping criteria
            if abs(new_value_table[state] - value_table[state]) > epsilon:
                value_table_stable = False
        value_table = new_value_table.copy()

        if value_table_stable:
            break

        # Policy improvement
        for state in range(env.observation_space.n):
            if state == goal_state:
                continue
            best_action = None
            # Negative infinity for initialization
            best_value = -float('inf')
            for action in range(env.action_space.n):
                # Check for valid action before accessing reward
                if len(env.P[state][action]) > 0:
                    value = 0.0
                    for prob, next_state, reward, _ in env.P[state][action]:
                        value += prob * (reward + discount_factor * value_table[next_state])
                    if value > best_value:
                        best_value = value
                        best_action = action
                else:
                    # Handle states with missing actions
                    pass
                # Update policy with the best action for this state
            policy[state] = best_action  

    return policy, value_table

# Create FrozenLake environment
env = gym.make('FrozenLake-v1')

# Run policy iteration
policy, value_table = policy_iteration(env)

# Print results
print("Optimal Policy:")
for state in range(env.observation_space.n):
    print(f"State: {state}, Action: {policy[state]}")

print("\nValue Table:")
print(value_table)

# Close the environment
env.close()
