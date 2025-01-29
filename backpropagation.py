import numpy as np

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Initialize the training set
# Input data: Each row is a training example, with 2 features
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Desired outputs: XOR logic (for this example)
y = np.array([[0], [1], [1], [0]])

# Set random seed for reproducibility
np.random.seed(42)

# Initialize weights with random values
input_layer_size = 2  # Number of input neurons (for 2 features)
hidden_layer_size = 2 # Number of hidden neurons
output_layer_size = 1 # Number of output neurons (binary output)

# Randomly initialize weights for input to hidden layer
w_input_hidden = np.random.uniform(-1, 1, (input_layer_size, hidden_layer_size))

# Randomly initialize weights for hidden to output layer
w_hidden_output = np.random.uniform(-1, 1, (hidden_layer_size, output_layer_size))

# Training parameters
learning_rate = 0.1
epochs = 10000

# Training loop (backpropagation)
for epoch in range(epochs):
    # Forward pass
    # Input to hidden layer
    hidden_layer_input = np.dot(X, w_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)

    # Hidden to output layer
    output_layer_input = np.dot(hidden_layer_output, w_hidden_output)
    output_layer_output = sigmoid(output_layer_input)

    # Calculate the error (difference between desired and actual output)
    error = y - output_layer_output

    # Backpropagation (calculating gradients and updating weights)
    
    # Calculate the gradient for output layer
    output_layer_delta = error * sigmoid_derivative(output_layer_output)

    # Calculate the gradient for hidden layer
    hidden_layer_error = output_layer_delta.dot(w_hidden_output.T)
    hidden_layer_delta = hidden_layer_error * sigmoid_derivative(hidden_layer_output)

    # Update weights using the gradient descent rule
    w_hidden_output += hidden_layer_output.T.dot(output_layer_delta) * learning_rate
    w_input_hidden += X.T.dot(hidden_layer_delta) * learning_rate

    # Print error every 1000 epochs
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Error: {np.mean(np.abs(error))}")

# Test the trained network
print("\nFinal output after training:")
print(output_layer_output)
