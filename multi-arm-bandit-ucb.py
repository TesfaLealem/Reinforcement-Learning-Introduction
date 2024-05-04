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

  def choose_action(self, c):
      # Ensure at least 1 count for each arm
    safe_counts = np.maximum(self.counts, 1)  
    upper_bounds = self.values + c * np.sqrt(np.log(self.counts.sum()) / safe_counts)
    return np.argmax(upper_bounds)

def main():
  # Define bandit parameters
  num_arms = 10
  # Random means for each arm
  means = np.random.rand(num_arms)  
  # Fixed variance for simplicity
  variances = np.ones(num_arms)  

  # Create bandit object
  bandit = Bandit(num_arms, means, variances)

  # Set number of time steps and UCB constant
  num_steps = 1000
  # Exploration constant (adjust for more/less exploration)
  c = 5  

  # Run simulation
  total_reward = 0
  for _ in range(num_steps):
    arm = bandit.choose_action(c)
    reward = bandit.pull(arm)
    total_reward += reward

  # Print results
  print(f"Total reward: {total_reward}")
  print(f"Average reward per step: {total_reward / num_steps}")

if __name__ == "__main__":
  main()
