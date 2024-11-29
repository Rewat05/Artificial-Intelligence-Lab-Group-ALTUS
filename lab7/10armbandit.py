import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

class NonStationaryBandit:
    def __init__(self, n_arms: int = 10, mean_init: float = 0.0, std_dev: float = 0.01):
        """
        Initialize non-stationary bandit with random walk of mean rewards
        
        Args:
            n_arms: Number of arms (default: 10)
            mean_init: Initial mean reward for all arms
            std_dev: Standard deviation for random walk step
        """
        self.n_arms = n_arms
        self.std_dev = std_dev
        # Initialize all arms with the same mean reward
        self.true_means = np.full(n_arms, mean_init)
        # Track mean history for visualization
        self.mean_history = [self.true_means.copy()]
        
    def pull(self, action: int) -> float:
        """
        Pull an arm and get reward
        
        Args:
            action: Arm to pull (0 to n_arms-1)
        Returns:
            float: Reward from the selected arm
        """
        if not (0 <= action < self.n_arms):
            raise ValueError(f"Action must be between 0 and {self.n_arms-1}")
        
        # Get reward from current mean with unit variance noise
        reward = np.random.normal(self.true_means[action], 1.0)
        
        # Update all means with random walk
        self.true_means += np.random.normal(0, self.std_dev, self.n_arms)
        self.mean_history.append(self.true_means.copy())
        
        return reward

class EpsilonGreedyAgent:
    def __init__(self, n_arms: int, epsilon: float = 0.1, step_size: float = 0.1):
        """
        Initialize epsilon-greedy agent with constant step-size
        
        Args:
            n_arms: Number of arms
            epsilon: Exploration probability
            step_size: Constant step size for updates (alpha)
        """
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.step_size = step_size
        
        # Initialize estimates and counts
        self.q_values = np.zeros(n_arms)
        self.action_counts = np.zeros(n_arms)
        
        # Track history for visualization
        self.q_history = [self.q_values.copy()]
        
    def select_action(self) -> int:
        """Select action using epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)
        return np.argmax(self.q_values)
    
    def update(self, action: int, reward: float):
        """
        Update value estimates using constant step-size
        
        Args:
            action: Action that was taken
            reward: Reward received
        """
        self.action_counts[action] += 1
        
        # Update using constant step-size
        self.q_values[action] += self.step_size * (reward - self.q_values[action])
        self.q_history.append(self.q_values.copy())

def run_experiment(n_steps: int = 10000, 
                  epsilon: float = 0.1, 
                  step_size: float = 0.1) -> Tuple[List[float], NonStationaryBandit, EpsilonGreedyAgent]:
    """
    Run non-stationary bandit experiment
    
    Args:
        n_steps: Number of time steps
        epsilon: Exploration probability
        step_size: Constant step size for updates
    
    Returns:
        Tuple of (rewards, bandit, agent)
    """
    bandit = NonStationaryBandit()
    agent = EpsilonGreedyAgent(10, epsilon, step_size)
    rewards = []
    
    for _ in range(n_steps):
        action = agent.select_action()
        reward = bandit.pull(action)
        agent.update(action, reward)
        rewards.append(reward)
    
    return rewards, bandit, agent

def plot_results(rewards: List[float], 
                bandit: NonStationaryBandit, 
                agent: EpsilonGreedyAgent,
                window_size: int = 100):
    """
    Plot results of the experiment
    
    Args:
        rewards: List of rewards received
        bandit: NonStationaryBandit instance
        agent: EpsilonGreedyAgent instance
        window_size: Window size for moving average
    """
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Moving average of rewards
    plt.subplot(2, 1, 1)
    moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    plt.plot(moving_avg)
    plt.title(f'Average Reward over Time (Window Size: {window_size})')
    plt.xlabel('Time Step')
    plt.ylabel('Average Reward')
    
    # Plot 2: True means vs Estimated values
    plt.subplot(2, 1, 2)
    mean_history = np.array(bandit.mean_history)
    q_history = np.array(agent.q_history)
    
    for arm in range(bandit.n_arms):
        plt.plot(mean_history[:, arm], '--', alpha=0.5, label=f'True Mean {arm}')
        plt.plot(q_history[:, arm], alpha=0.5, label=f'Estimated {arm}')
    
    plt.title('True Means vs Estimated Values')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Run experiment
    rewards, bandit, agent = run_experiment(
        n_steps=10000,
        epsilon=0.1,
        step_size=0.1
    )
    
    # Plot results
    plot_results(rewards, bandit, agent)
    
    # Print final statistics
    print("\nFinal Statistics:")
    print(f"Average reward: {np.mean(rewards):.3f}")
    print(f"Action counts: {agent.action_counts}")
    print(f"Final Q-values: {agent.q_values}")