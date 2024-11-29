import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

class NonStationaryBandit:
    def __init__(self, n_arms: int = 10, mean_init: float = 0.0, std_dev: float = 0.01):
        """Non-stationary bandit with random walk of mean rewards"""
        self.n_arms = n_arms
        self.std_dev = std_dev
        self.true_means = np.full(n_arms, mean_init)
        self.mean_history = [self.true_means.copy()]
        
    def pull(self, action: int) -> float:
        """Pull an arm and get reward"""
        if not (0 <= action < self.n_arms):
            raise ValueError(f"Action must be between 0 and {self.n_arms-1}")
        
        reward = np.random.normal(self.true_means[action], 1.0)
        
        # Random walk of mean rewards
        self.true_means += np.random.normal(0, self.std_dev, self.n_arms)
        self.mean_history.append(self.true_means.copy())
        
        return reward

class StandardEpsilonGreedy:
    """Standard epsilon-greedy with sample averaging"""
    def __init__(self, n_arms: int, epsilon: float = 0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.q_values = np.zeros(n_arms)
        self.action_counts = np.zeros(n_arms)
        self.q_history = [self.q_values.copy()]
        
    def select_action(self) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)
        return np.argmax(self.q_values)
    
    def update(self, action: int, reward: float):
        """Update using sample averaging"""
        self.action_counts[action] += 1
        n = self.action_counts[action]
        self.q_values[action] += (reward - self.q_values[action]) / n
        self.q_history.append(self.q_values.copy())

class ModifiedEpsilonGreedy:
    """Modified epsilon-greedy with constant step size for non-stationary problems"""
    def __init__(self, n_arms: int, epsilon: float = 0.1, step_size: float = 0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.step_size = step_size
        self.q_values = np.zeros(n_arms)
        self.action_counts = np.zeros(n_arms)
        self.q_history = [self.q_values.copy()]
        
    def select_action(self) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)
        return np.argmax(self.q_values)
    
    def update(self, action: int, reward: float):
        """Update using constant step size"""
        self.action_counts[action] += 1
        self.q_values[action] += self.step_size * (reward - self.q_values[action])
        self.q_history.append(self.q_values.copy())

def run_comparison(n_steps: int = 10000, 
                  epsilon: float = 0.1, 
                  step_size: float = 0.1) -> Dict:
    """Run comparison between standard and modified agents"""
    # Initialize bandits and agents
    bandit1 = NonStationaryBandit()
    bandit2 = NonStationaryBandit()
    standard_agent = StandardEpsilonGreedy(10, epsilon)
    modified_agent = ModifiedEpsilonGreedy(10, epsilon, step_size)
    
    # Track results
    results = {
        'standard_rewards': [],
        'modified_rewards': [],
        'standard_optimal_actions': [],
        'modified_optimal_actions': [],
        'bandit1': bandit1,
        'bandit2': bandit2,
        'standard_agent': standard_agent,
        'modified_agent': modified_agent
    }
    
    # Run experiments
    for _ in range(n_steps):
        # Standard agent
        action1 = standard_agent.select_action()
        reward1 = bandit1.pull(action1)
        standard_agent.update(action1, reward1)
        results['standard_rewards'].append(reward1)
        results['standard_optimal_actions'].append(
            action1 == np.argmax(bandit1.true_means)
        )
        
        # Modified agent
        action2 = modified_agent.select_action()
        reward2 = bandit2.pull(action2)
        modified_agent.update(action2, reward2)
        results['modified_rewards'].append(reward2)
        results['modified_optimal_actions'].append(
            action2 == np.argmax(bandit2.true_means)
        )
    
    return results

def plot_comparison_results(results: Dict, window_size: int = 200):
    """Plot comparative results"""
    plt.figure(figsize=(15, 15))
    
    # Plot 1: Average rewards
    plt.subplot(3, 1, 1)
    standard_avg = np.convolve(results['standard_rewards'], 
                              np.ones(window_size)/window_size, 
                              mode='valid')
    modified_avg = np.convolve(results['modified_rewards'], 
                              np.ones(window_size)/window_size, 
                              mode='valid')
    
    plt.plot(standard_avg, label='Standard ε-greedy', alpha=0.8)
    plt.plot(modified_avg, label='Modified ε-greedy', alpha=0.8)
    plt.title(f'Average Reward over Time (Window Size: {window_size})')
    plt.xlabel('Time Step')
    plt.ylabel('Average Reward')
    plt.legend()
    
    # Plot 2: Optimal action percentage
    plt.subplot(3, 1, 2)
    standard_optimal = np.convolve(results['standard_optimal_actions'], 
                                 np.ones(window_size)/window_size, 
                                 mode='valid')
    modified_optimal = np.convolve(results['modified_optimal_actions'], 
                                 np.ones(window_size)/window_size, 
                                 mode='valid')
    
    plt.plot(standard_optimal, label='Standard ε-greedy', alpha=0.8)
    plt.plot(modified_optimal, label='Modified ε-greedy', alpha=0.8)
    plt.title('Optimal Action Selection Percentage')
    plt.xlabel('Time Step')
    plt.ylabel('% Optimal Actions')
    plt.legend()
    
    # Plot 3: True means vs Estimated values for modified agent
    plt.subplot(3, 1, 3)
    mean_history = np.array(results['bandit2'].mean_history)
    q_history = np.array(results['modified_agent'].q_history)
    
    for arm in range(10):
        plt.plot(mean_history[:, arm], '--', alpha=0.3, 
                label=f'True Mean {arm}' if arm == 0 else None)
        plt.plot(q_history[:, arm], alpha=0.3, 
                label=f'Estimated {arm}' if arm == 0 else None)
    
    plt.title('Modified Agent: True Means vs Estimated Values')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def print_comparison_stats(results: Dict):
    """Print comparative statistics"""
    print("\nComparison Statistics:")
    print("\nStandard ε-greedy:")
    print(f"Average reward: {np.mean(results['standard_rewards']):.3f}")
    print(f"Optimal action percentage: {np.mean(results['standard_optimal_actions'])*100:.1f}%")
    print(f"Action counts: {results['standard_agent'].action_counts}")
    
    print("\nModified ε-greedy:")
    print(f"Average reward: {np.mean(results['modified_rewards']):.3f}")
    print(f"Optimal action percentage: {np.mean(results['modified_optimal_actions'])*100:.1f}%")
    print(f"Action counts: {results['modified_agent'].action_counts}")

if __name__ == "__main__":
    # Run comparison
    results = run_comparison(
        n_steps=10000,
        epsilon=0.1,
        step_size=0.1
    )
    
    # Plot and print results
    plot_comparison_results(results)
    print_comparison_stats(results)