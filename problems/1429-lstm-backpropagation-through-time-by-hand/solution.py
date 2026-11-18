import numpy as np


def _sigmoid(z):
    z = np.asarray(z)
    out = np.empty_like(z, dtype=np.result_type(z, np.float64))
    mask = z >= 0
    out[mask] = 1.0 / (1.0 + np.exp(-z[mask]))
    ez = np.exp(z[~mask])
    out[~mask] = ez / (1.0 + ez)
    return out


def lstm_forward(X, Wx, Wh, b, h0, c0):
    T, d = X.shape
    h = h0.shape[0]

    H = np.empty((T, h), dtype=np.result_type(X, Wx, Wh, b, h0, c0))
    C = np.empty_like(H)
    cache = []

    hp = h0.copy()
    cp = c0.copy()

    for t in range(T):
        x = X[t]
        z = Wx @ x + Wh @ hp + b

        zi, zf, zo, zg = np.split(z, 4)
        i = _sigmoid(zi)
        f = _sigmoid(zf)
        o = _sigmoid(zo)
        g = np.tanh(zg)

        c = f * cp + i * g
        ht = o * np.tanh(c)

        cache.append((x, hp, cp, i, f, o, g, c))
        H[t] = ht
        C[t] = c

        hp = ht
        cp = c

    return H, C, cache


def lstm_backward(dH, cache, Wx, Wh):
    T, h = dH.shape
    d = Wx.shape[1]

    dWx = np.zeros_like(Wx, dtype=np.result_type(dH, Wx))
    dWh = np.zeros_like(Wh, dtype=np.result_type(dH, Wh))
    db = np.zeros(4 * h, dtype=np.result_type(dH, Wx))

    dh_next = np.zeros(h, dtype=db.dtype)
    dc_next = np.zeros(h, dtype=db.dtype)

    for t in range(T - 1, -1, -1):
        x, hp, cp, i, f, o, g, c = cache[t]

        dh = dH[t] + dh_next
        tc = np.tanh(c)

        # Gradient through h = o * tanh(c)
        do = dh * tc
        dc = dh * o * (1.0 - tc * tc) + dc_next

        # Gradients through c = f*c_prev + i*g
        df = dc * cp
        dc_prev = dc * f
        di = dc * g
        dg = dc * i

        # Gradients through gate activations
        dzi = di * i * (1.0 - i)
        dzf = df * f * (1.0 - f)
        dzo = do * o * (1.0 - o)
        dzg = dg * (1.0 - g * g)

        dz = np.concatenate((dzi, dzf, dzo, dzg))

        dWx += np.outer(dz, x)
        dWh += np.outer(dz, hp)
        db += dz

        dh_next = Wh.T @ dz
        dc_next = dc_prev

    return dWx, dWh, db, dh_next, dc_next


def sequence_loss(X, Wx, Wh, b, targets):
    H, C, cache = lstm_forward(
        X, Wx, Wh, b,
        np.zeros(Wh.shape[1], dtype=Wx.dtype),
        np.zeros(Wh.shape[1], dtype=Wx.dtype)
    )

    diff = H - targets
    loss = np.mean(diff * diff)
    dH = (2.0 / diff.size) * diff

    return loss, dH