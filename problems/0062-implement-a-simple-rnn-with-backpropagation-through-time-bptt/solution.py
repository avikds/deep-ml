import numpy as np

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size

        self.W_xh = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W_hy = np.random.randn(output_size, hidden_size) * 0.01

        self.b_h = np.zeros((hidden_size, 1))
        self.b_y = np.zeros((output_size, 1))

    def forward(self, x):
        x = np.asarray(x, dtype=float)
        T = x.shape[0]

        hidden_states = np.zeros((T + 1, self.hidden_size, 1))
        outputs = []

        for t in range(T):
            x_t = x[t].reshape(-1, 1)

            hidden_states[t + 1] = np.tanh(
                self.W_xh @ x_t +
                self.W_hh @ hidden_states[t] +
                self.b_h
            )

            output_t = self.W_hy @ hidden_states[t + 1] + self.b_y
            outputs.append(output_t)

        self._hidden_states = hidden_states

        return np.array(outputs)

    def backward(self, x, y, learning_rate):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        outputs = self.forward(x)
        T = x.shape[0]

        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)

        dh_next = np.zeros((self.hidden_size, 1))

        for t in range(T - 1, -1, -1):
            x_t = x[t].reshape(-1, 1)
            y_t = y[t].reshape(-1, 1)

            h_t = self._hidden_states[t + 1]
            h_prev = self._hidden_states[t]

            # Loss = 1/2 * (output - target)^2
            dy = outputs[t] - y_t

            dW_hy += dy @ h_t.T
            db_y += dy

            dh = self.W_hy.T @ dy + dh_next

            # Derivative of tanh
            dz = dh * (1.0 - h_t ** 2)

            dW_xh += dz @ x_t.T
            dW_hh += dz @ h_prev.T
            db_h += dz

            dh_next = self.W_hh.T @ dz

        self.W_xh -= learning_rate * dW_xh
        self.W_hh -= learning_rate * dW_hh
        self.W_hy -= learning_rate * dW_hy
        self.b_h -= learning_rate * db_h
        self.b_y -= learning_rate * db_y