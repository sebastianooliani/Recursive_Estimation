###########################################
# Realized by: Sebastiano Oliani
# 10/05/2023
###########################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy

if __name__ == "__main__":
    mu = 3.9
    alpha = 0.07
    a = 0.6
    b = 0.15

    N = 100

    x = np.zeros((N, 1))
    y = np.zeros_like(x)
    z_x = np.zeros_like(x)
    z_y = np.zeros_like(x)

    # generate dynamics
    x[0] = np.random.uniform(0, 1)
    y[0] = np.random.uniform(0, 1)

    for i in range(1, N):
        wx = np.random.uniform(0, b)
        wy = np.random.uniform(0, b)
        z_x[i-1] = x[i-1] * (1 - wx)
        z_y[i-1] = y[i-1] * (1 - wy)

        vy = np.random.uniform(0, a)
        lx = mu * x[i-1] * (1 - x[i-1])
        vx = np.random.uniform(0, a)
        ly = mu * y[i-1] * (1 - y[i-1])

        x[i] = (1 - alpha) * lx + (1 - vy) * alpha * ly
        y[i] = (1 - alpha) * ly + (1 - vx) * alpha * lx

    # mean of a uniform RV between a and b = ((b+a))/2
    # variance of a uniform RV between a and b = ((b-a)^2)/12
    mean = a/2
    var = a/12
   
    xp = np.zeros((N, 1))
    yp = np.zeros_like(x)
    xm = np.zeros_like(x)
    ym = np.zeros_like(x)

    Ppx = np.zeros_like(x)
    Pmx = np.zeros_like(x)
    Ppy = np.zeros_like(x)
    Pmy = np.zeros_like(x)

    ## EXTENDED KALMAN FILTER
    xp[0] = mean
    yp[0] = mean
    xm[0] = mean
    ym[0] = mean
    Ppx[0] = var
    Pmx[0] = var
    Ppy[0] = var
    Pmy[0] = var

    for i in range(1, N):
        # prior update
        A = (1 - alpha) * (mu - mu * 2 * xm[i-1]) + (1 - (0 + mean)) * alpha * (mu - mu * 2 * ym[i-1])
        L = - alpha * (mu * ym[i-1] * (1 - ym[i-1]))

        lx = mu * xm[i-1] * (1 - xm[i-1])
        ly = mu * ym[i-1] * (1 - ym[i-1])
        xp[i] = (1 - alpha) * lx + (1 - vy) * alpha * ly
        Ppx[i] = A * Pmx[i-1] * A + L * var * L

        A = (1 - alpha) * (mu - mu * 2 * ym[i-1]) + (1 - (0 + mean)) * alpha * (mu - mu * 2 * xm[i-1])
        L = - alpha * (mu * xm[i-1] * (1 - xm[i-1]))

        yp[i] = (1 - alpha) * ly + (1 - vy) * alpha * lx
        Ppy[i] = A * Pmy[i-1] * A + L * var * L

        # measurement update
