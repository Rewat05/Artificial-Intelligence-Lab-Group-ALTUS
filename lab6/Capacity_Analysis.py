import numpy as np

class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))

    def train(self, patterns):
        """Train the network with patterns."""
        for pattern in patterns:
            bipolar_pattern = 2 * pattern - 1  # Convert binary (0,1) to bipolar (-1,1)
            self.weights += np.outer(bipolar_pattern, bipolar_pattern)
        np.fill_diagonal(self.weights, 0)  # No self-loops

    def recall(self, pattern, steps=5):
        """Recall a pattern."""
        state = 2 * pattern - 1  # Convert to bipolar
        for _ in range(steps):
            for i in range(self.size):
                raw_sum = np.dot(self.weights[i], state)
                state[i] = 1 if raw_sum > 0 else -1
        return (state + 1) // 2  # Convert back to binary (0,1)

    def test_capacity(self, patterns):
        """Test if the network can recall all patterns correctly."""
        self.train(patterns)
        success = 0
        for pattern in patterns:
            noisy_pattern = pattern.copy()
            recalled_pattern = self.recall(noisy_pattern)
            if np.array_equal(recalled_pattern, pattern):
                success += 1
        return success / len(patterns)  # Fraction of successfully recalled patterns

# Parameters
N = 100  # Size of the network (10x10)
P_max_theoretical = int(0.15 * N)  # Theoretical capacity

# Generate random binary patterns
patterns = [np.random.randint(0, 2, N) for _ in range(P_max_theoretical)]

# Test the capacity
hopfield = HopfieldNetwork(N)
recall_rate = hopfield.test_capacity(patterns)

print(f"Theoretical Capacity (P_max): {P_max_theoretical} patterns")
print(f"Recall Rate with {P_max_theoretical} patterns: {recall_rate * 100:.2f}%")