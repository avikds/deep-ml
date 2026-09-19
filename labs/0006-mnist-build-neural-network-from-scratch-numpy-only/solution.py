import numpy as np

class NeuralNetwork:
    '''
    Build a neural network from scratch using only NumPy.
    Required architecture: 784 → 128 (ReLU) → 10 (Softmax)
    '''
    def __init__(self, input_size=784, hidden_size=128, output_size=10, lr=0.01):
        self.lr = lr

        # He initialization for ReLU layer
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))

        # Xavier initialization for output layer
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))

        # Cache for backpropagation
        self.z1 = None
        self.a1 = None
        self.z2 = None

    def forward(self, X):
        '''
        Forward pass.

        X: (N, 784)
        Returns: probabilities (N, 10)
        '''
        # First linear layer
        self.z1 = X @ self.W1 + self.b1

        # ReLU
        self.a1 = np.maximum(0, self.z1)

        # Second linear layer
        self.z2 = self.a1 @ self.W2 + self.b2

        # Numerically stable softmax
        shifted = self.z2 - np.max(self.z2, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        return probs

    def backward(self, X, y, probs):
        '''
        Backward pass and parameter update.

        Returns:
            loss: scalar cross-entropy loss
        '''
        N = X.shape[0]

        # ---------------------------------------------------------
        # 1. Cross-entropy loss
        # ---------------------------------------------------------
        # Clip probabilities to avoid log(0)
        correct_probs = np.clip(probs[np.arange(N), y], 1e-12, 1.0)
        loss = -np.mean(np.log(correct_probs))

        # ---------------------------------------------------------
        # 2. Gradient through Softmax + Cross Entropy
        # ---------------------------------------------------------
        dz2 = probs.copy()
        dz2[np.arange(N), y] -= 1
        dz2 /= N

        # ---------------------------------------------------------
        # 3. Backprop through second linear layer
        # z2 = a1 @ W2 + b2
        # ---------------------------------------------------------
        dW2 = self.a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ self.W2.T

        # ---------------------------------------------------------
        # 4. Backprop through ReLU
        # ReLU'(z) = 1 if z > 0, otherwise 0
        # ---------------------------------------------------------
        dz1 = da1 * (self.z1 > 0)

        # ---------------------------------------------------------
        # 5. Backprop through first linear layer
        # z1 = X @ W1 + b1
        # ---------------------------------------------------------
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # ---------------------------------------------------------
        # 6. Gradient descent parameter updates
        # ---------------------------------------------------------
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

        return loss

    def train_step(self, X, y):
        '''
        Complete training step: forward + backward + update.
        '''
        probs = self.forward(X)
        loss = self.backward(X, y, probs)
        return loss

    def predict(self, X):
        '''
        Predict class labels.
        '''
        probs = self.forward(X)
        return np.argmax(probs, axis=1)