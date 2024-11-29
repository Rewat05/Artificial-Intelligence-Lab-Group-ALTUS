import numpy as np
from typing import Tuple, List

class BinaryBanditA:
    def __init__(self):
        """Implementation of first binary bandit with p = [0.1, 0.2]"""
        self.p = [0.1, 0.2]
    
    def pull(self, action: int) -> int:
        """
        Args:
            action: 1 or 2 (using 1-based indexing as per problem)
        Returns:
            1 (success) or 0 (failure)
        """
        if not (action in [1, 2]):
            raise ValueError("Action must be 1 or 2")
        
        if np.random.random() < self.p[action-1]:
            return 1
        return 0

class BinaryBanditB:
    def __init__(self):
        """Implementation of second binary bandit with p = [0.8, 0.9]"""
        self.p = [0.8, 0.9]
    
    def pull(self, action: int) -> int:
        """
        Args:
            action: 1 or 2 (using 1-based indexing as per problem)
        Returns:
            1 (success) or 0 (failure)
        """
        if not (action in [1, 2]):
            raise ValueError("Action must be 1 or 2")
            
        if np.random.random() < self.p[action-1]:
            return 1
        return 0

class EpsilonGreedy:
    def __init__(self, epsilon: float = 0.1):
        """
        Initialize epsilon-greedy algorithm
        
        Args:
            epsilon: Exploration probability (0 to 1)
        """
        self.epsilon = epsilon
        # Using 1-based indexing, index 0 will be unused
        self.counts = [0, 0, 0]  # Number of times each action was taken
        self.values = [0.0, 0.0, 0.0]  # Estimated value for each action
    
    def select_action(self) -> int:
        """
        Select which action to take using epsilon-greedy strategy
        
        Returns:
            int: Selected action (1 or 2)
        """
        if np.random.random() < self.epsilon:
            # Explore: choose random action
            return np.random.choice([1, 2])
        else:
            # Exploit: choose best action (add 1 for 1-based indexing)
            return np.argmax(self.values[1:]) + 1
    
    def update(self, action: int, reward: int):
        """
        Update estimated values based on received reward
        
        Args:
            action: The action that was taken (1 or 2)
            reward: The reward received (0 or 1)
        """
        self.counts[action] += 1
        n = self.counts[action]
        value = self.values[action]
        self.values[action] = ((n - 1) / n) * value + (1 / n) * reward

def run_experiment(n_steps: int = 1000, epsilon: float = 0.1) -> None:
    """
    Run the bandit experiment
    
    Args:
        n_steps: Number of steps to run
        epsilon: Exploration probability
    """
    # Initialize bandits and agent
    bandit_a = BinaryBanditA()
    bandit_b = BinaryBanditB()
    agent = EpsilonGreedy(epsilon)
    
    # Track results
    total_reward = 0
    rewards_history = []
    action_history = []
    
    for step in range(n_steps):
        # Select action
        action = agent.select_action()
        action_history.append(action)
        
        # Get reward from selected bandit
        reward = bandit_a.pull(action) if isinstance(action, int) else 0
        reward_b = bandit_b.pull(action) if isinstance(action, int) else 0
        
        # Take maximum reward between the two bandits
        reward = max(reward, reward_b)
        
        # Update agent's estimates
        agent.update(action, reward)
        
        # Track results
        total_reward += reward
        rewards_history.append(total_reward / (step + 1))
    
    # Print results
    print(f"\nResults after {n_steps} steps:")
    print(f"Epsilon value: {epsilon}")
    print(f"Final action values: Action 1: {agent.values[1]:.3f}, Action 2: {agent.values[2]:.3f}")
    print(f"Action counts: Action 1: {agent.counts[1]}, Action 2: {agent.counts[2]}")
    print(f"Final average reward: {rewards_history[-1]:.3f}")
    
    # Calculate action percentages
    action1_percentage = (agent.counts[1] / n_steps) * 100
    print(f"\nAction selection percentages:")
    print(f"Action 1: {action1_percentage:.1f}%")
    print(f"Action 2: {100-action1_percentage:.1f}%")

if __name__ == "__main__":
    # Run the experiment
    run_experiment(n_steps=1000, epsilon=0.1)