import numpy as np
import matplotlib.pyplot as plt

# in the figure 1 represnts blank space and 0 represents dark square

class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))  # Initialize weight matrix

    def train(self, patterns):
        """Train the Hopfield network with the given binary patterns."""
        for pattern in patterns:
            # Ensure the pattern is binary (-1, 1)
            bipolar_pattern = 2 * pattern - 1
            self.weights += np.outer(bipolar_pattern, bipolar_pattern)
        
        # Zero out the diagonal (no self-connections)
        np.fill_diagonal(self.weights, 0)

    def recall(self, pattern, steps=5):
        """Recall a pattern from the network using asynchronous updates."""
        state = pattern.copy()
        for _ in range(steps):
            for i in range(self.size):
                raw_sum = np.dot(self.weights[i], state)
                state[i] = 1 if raw_sum > 0 else -1
        return (state + 1) // 2  # Convert back to binary (0, 1)

    def energy(self, state):
        """Calculate the energy of the current state."""
        bipolar_state = 2 * state - 1
        return -0.5 * np.dot(bipolar_state.T, np.dot(self.weights, bipolar_state))

# Define a 10x10 Hopfield network
size = 100  # For a 10x10 grid, size is 100
hopfield = HopfieldNetwork(size)

# Create binary patterns (each pattern is 10x10 flattened into a 1D array)
patterns = [
    np.random.randint(0, 2, size),  # Random binary pattern
    np.random.randint(0, 2, size),  # Another random binary pattern
]

# Train the network
hopfield.train(patterns)

# Test the network
test_pattern = patterns[0].copy()
# Introduce noise to the test pattern
noise_indices = np.random.choice(size, size // 10, replace=False)
test_pattern[noise_indices] = 1 - test_pattern[noise_indices]  # Flip bits

# Recall the pattern
recalled_pattern = hopfield.recall(test_pattern)

# Function to display and print the patterns
def display_and_print_pattern(pattern, title):
    """Display the pattern as an image and print it as a matrix."""
    pattern_matrix = pattern.reshape(10, 10)
    print(f"{title} (Binary Matrix):")
    print(pattern_matrix)
    print("\n")
    plt.imshow(pattern_matrix, cmap="gray", interpolation="nearest")
    plt.title(title)
    plt.axis("off")

# Plot and print the patterns
plt.figure(figsize=(10, 4))

# Original pattern
plt.subplot(1, 3, 1)
display_and_print_pattern(patterns[0], "Original Pattern")

# Noisy input pattern
plt.subplot(1, 3, 2)
display_and_print_pattern(test_pattern, "Noisy Input Pattern")

# Recalled pattern
plt.subplot(1, 3, 3)
display_and_print_pattern(recalled_pattern, "Recalled Pattern")

plt.tight_layout()
plt.show()