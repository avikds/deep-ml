import numpy as np

def optimizer_step(param, grad, state, lr):
    '''
    Update a parameter using Adam.

    Args:
        param: numpy array - current parameter values
        grad: numpy array - gradient of loss w.r.t. param
        state: dict - persistent optimizer state
        lr: float - learning rate

    Returns:
        new_param: numpy array - updated parameter
        state: updated optimizer state
    '''
    # Adam hyperparameters
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    # Initialize state on first call
    if 'step' not in state:
        state['step'] = 0
        state['m'] = np.zeros_like(param)
        state['v'] = np.zeros_like(param)

    # Increment timestep
    state['step'] += 1
    t = state['step']

    # Update biased first and second moment estimates
    state['m'] = beta1 * state['m'] + (1 - beta1) * grad
    state['v'] = beta2 * state['v'] + (1 - beta2) * (grad * grad)

    # Bias correction
    m_hat = state['m'] / (1 - beta1 ** t)
    v_hat = state['v'] / (1 - beta2 ** t)

    # Adam parameter update
    new_param = param - lr * m_hat / (np.sqrt(v_hat) + eps)

    return new_param, state