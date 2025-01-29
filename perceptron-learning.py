import numpy as np

# Perceptron activation function (step function)
def step_function(x):
    return 1 if x >= 0 else 0

# Perceptron Model
class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        # Initialize weights and bias
        self.weights = np.random.randn(input_size)  # Randomly initialize weights
        self.bias = np.random.randn()  # Random bias
        self.learning_rate = learning_rate

    # Forward pass: Calculate the output of the perceptron
    def predict(self, inputs):
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        return step_function(weighted_sum)

    # Training the perceptron using the perceptron learning rule
    def train(self, training_inputs, training_labels, epochs):
        for epoch in range(epochs):
            total_error = 0
            for inputs, label in zip(training_inputs, training_labels):
                # Predict the output for the given input
                prediction = self.predict(inputs)
                # Calculate the error
                error = label - prediction
                total_error += abs(error)
                # Update weights and bias
                self.weights += self.learning_rate * error * inputs
                self.bias += self.learning_rate * error
            # Print total error for each epoch
            if epoch % 1000 == 0:
                print(f'Epoch {epoch}, Total Error: {total_error}')

# Define the training dataset for the AND gate
# Inputs: (X0, X1)
X = np.array([
    [0, 0],  # Input 1: [0, 0]
    [0, 1],  # Input 2: [0, 1]
    [1, 0],  # Input 3: [1, 0]
    [1, 1]   # Input 4: [1, 1]
])

# Desired outputs for AND gate
y = np.array([0, 0, 0, 1])  # Corresponding outputs for AND gate

# Create a Perceptron instance
perceptron = Perceptron(input_size=2, learning_rate=0.1)

# Train the Perceptron for 10000 epochs
perceptron.train(X, y, epochs=10000)

# Test the perceptron after training
print("\nTesting trained Perceptron:")
for x in X:
    print(f"Input: {x} => Predicted Output: {perceptron.predict(x)}")
