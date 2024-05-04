import numpy as np

class Bandit:
  def __init__(self, num_arms, means, variances):
    self.num_arms = num_arms
    self.means = means
    self.variances = variances
    # Count of times each arm is pulled
    self.counts = np.zeros(num_arms) 
    # Estimated mean reward for each arm
    self.values = np.zeros(num_arms)  

  def pull(self, arm):
    reward = np.random.normal(self.means[arm], self.variances[arm])
    self.counts[arm] += 1
    self.values[arm] = (self.values[arm] * (self.counts[arm] - 1) + reward) / self.counts[arm]
    return reward

  def choose_action(self, epsilon):
      # Explore with probability epsilon
    if np.random.rand() < epsilon: 
      return np.random.randint(0, self.num_arms)
  # Exploit: choose arm with highest estimated reward
    else:  
      return np.argmax(self.values)

def main():
  # Define bandit parameters
  num_arms = 10
  # Random means for each arm
  means = np.random.rand(num_arms) 
  # Fixed variance for simplicity
  variances = np.ones(num_arms) 

  # Create bandit object
  bandit = Bandit(num_arms, means, variances)

  # Set number of time steps and epsilon value
  num_steps = 1000
  epsilon = 0.1

  # Run simulation
  total_reward = 0
  for _ in range(num_steps):
    arm = bandit.choose_action(epsilon)
    reward = bandit.pull(arm)
    total_reward += reward

  # Print results
  print(f"Total reward: {total_reward}")
  print(f"Average reward per step: {total_reward / num_steps}")

if __name__ == "__main__":
  main()
