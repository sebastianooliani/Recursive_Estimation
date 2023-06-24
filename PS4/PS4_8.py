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

    x = np.zeros((1, N))
    y = np.zeros_like(x)
    z_x = np.zeros_like(x)
    z_y = np.zeros_like(x)

    # generate dynamics
    x[0][0] = np.random.rand()
    y[0][0] = np.random.rand()

    for i in range(1, N):
        wx = np.random.uniform(0, b)
        wy = np.random.uniform(0, b)
        z_x[0][i-1] = x[0][i-1] * (1 - wx)
        z_y[0][i-1] = y[0][i-1] * (1 - wy)

        vy = np.random.uniform(0, a)
        lx = mu * x[0][i-1] * (1 - x[0][i-1])
        vx = np.random.uniform(0, a)
        ly = mu * y[0][i-1] * (1 - y[0][i-1])

        x[0][i] = (1 - alpha) * lx + (1 - vy) * alpha * ly
        y[0][i] = (1 - alpha) * ly + (1 - vx) * alpha * lx

    # mean of a uniform RV between a and b = ((b+a))/2
    # variance of a uniform RV between a and b = ((b-a)^2)/12
    mean = 1 / 2
    var = 1 / 12
   
    xp = np.zeros((1, N))
    yp = np.zeros_like(x)
    xm = np.zeros_like(x)
    ym = np.zeros_like(x)

    Pp = np.zeros((N, 2, 2))
    Pm = np.zeros_like(Pp)

    ## EXTENDED KALMAN FILTER
    xp[0][0] = mean
    yp[0][0] = mean
    xm[0][0] = mean
    ym[0][0] = mean
    Pp[0][0][0] = var
    Pp[0][1][1] = var
    Pm[0][0][0] = var
    Pm[0][1][1] = var

    A = np.zeros((2 , 2))
    L = np.zeros_like(A)
    H = np.eye(2) * (1 - b / 2)
    M = np.zeros_like(A)
    Q = np.eye(2) * a**2 / 12
    R = np.eye(2) * b**2 / 12

    for i in range(0, N-1):
        # prior update
        A[0][:] = [(1 - alpha) * (mu - mu * 2 * xm[0][i-1]), (1 - (0 + a / 2)) * alpha * (mu - mu * 2 * ym[0][i-1])]
        A[1][:] = [(1 - alpha) * (mu - mu * 2 * ym[0][i-1]), (1 - (0 + a / 2)) * alpha * (mu - mu * 2 * xm[0][i-1])]      
        L[0][:] = [0, - alpha * (mu * ym[0][i-1] * (1 - ym[0][i-1]))]    
        L[1][:] = [- alpha * (mu * xm[0][i-1] * (1 - xm[0][i-1])), 0]

        lx = mu * xm[0][i-1] * (1 - xm[0][i-1])
        ly = mu * ym[0][i-1] * (1 - ym[0][i-1])
        xp[0][i] = (1 - alpha) * lx + (1 - a / 2) * alpha * ly
        yp[0][i] = (1 - alpha) * ly + (1 - a / 2) * alpha * lx
        Pp[i+1, :, :] = A @ Pm[i, :, :] @ A.T + L @ Q @ L.T

        # measurement update
        M[0][0] = - xp[0][i]
        M[1][1] = - yp[0][i]

        K = Pp[i, :, :] @ H.T @ (H @ Pp[i, :, :] @ H.T + M @ R @ M.T) ** (-1)
        xm[0][i] = xp[0][i] + K[0, 0] * (z_x[0][i] - H[0, 0] * (xp[0][i] * (1 - b/2)))
        ym[0][i] = yp[0][i] + K[1, 0] * (z_y[0][i] - H[0, 1] * (yp[0][i] * (1 - b/2)))
        Pm[i, :, :] = (np.identity(2) - K @ H) @ Pp[i, :, :]

    # Plot results
    labels = ["$x$", "$y$", r"$\alpha$"]
    fig, axs = plt.subplots(2, 1)

    
    axs[0].plot(np.arange(0, N), x[0, :], label="Actual")
    axs[0].plot(np.arange(0, N), xm[0, :], label="Estimated")
    axs[0].set_xlabel("$k$")
    axs[0].set_ylabel(labels[0])
    axs[0].legend()

    axs[1].plot(np.arange(0, N), y[0, :], label="Actual")
    axs[1].plot(np.arange(0, N), ym[0, :], label="Estimated")
    axs[1].set_xlabel("$k$")
    axs[1].set_ylabel(labels[1])
    axs[1].legend()

    """if PART == "b":
        axs[2].plot(np.arange(0, N + 1), alpha * np.ones(T + 1), label="Actual")
        axs[2].plot(np.arange(0, N + 1), sm[2, :], label="Estimated")
        axs[2].set_xlabel("$k$")
        axs[2].set_ylabel(labels[2])
        axs[2].legend()"""

    plt.show()