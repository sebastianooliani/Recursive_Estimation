###########################################
# Realized by: Sebastiano Oliani
# 10/04/2023
###########################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy

if __name__ == "__main__":
    # Choose which part of the problem you would like to run:
    print("Choose part of the problem to run: ")
    PART = input()

    if (
        PART != "b"
        and PART != "c"
        and PART != "d"
        and PART != "e"
        and PART != "f"
        and PART != "g"
    ):
        raise ValueError('Invalid input. Please choose from "b" to "g".')
    
    alpha_1 = 0.1
    alpha_2 = 0.5
    alpha_3 = 0.2
    N = 1000

    if PART == "f":
        A = np.array([[1 - alpha_1, 0, 0], [0, 1 - alpha_2, 0], [alpha_1, alpha_2, 1 - alpha_3]])
    else:
        A = np.array([[1 - alpha_1, 0, 0], [0, 1, 0], [alpha_1, 0, 1 - alpha_3]])
    B = np.array([[0.5], [0.5], [0]])
    if PART == "c" or PART == "d":
        H = np.array([[0, 1, 0], [0, 0, 1]])
    else:
        H = np.array([[1, 0, 0], [0, 0, 1]])

    x0 = np.ones((3, 1)) * 5
    P = np.zeros((3, 3, N + 1))
    P[:, :, 0] = np.identity(3)
    Q = np.diag([1/40, 1/10, 1/5])
    R = np.identity(2) * 0.5

    if PART == "b":
        P_inf = scipy.linalg.solve_discrete_are(A.T, H.T, Q, R)
        print(P_inf)

    ## GENERATE DYNAMICS
    x = np.zeros((3, N + 1))
    x[:, 0] = np.squeeze(x0)
    z = np.zeros((2, N + 1))

    # Control input
    if PART == "d":
        u = 5 * np.abs(np.sin(np.arange(1, N + 1)))
        u = np.resize(u, (1, N))
    else:
        u = 5 * np.ones((1, N))

    w = np.random.randn(2, N + 1) * (0.5 ** 0.5)
    v = np.linalg.cholesky(Q) @ np.random.randn(3, N)

    for k in range(1, N+1):
        x[:, k] = A @ x[:, k-1] + B @ u[:, k-1] + v[:, k-1]
        z[:, k] = H @ x[:, k] + w[:, k]

    ## KALMAN FILTER IMPLEMENTATION
    x_p = np.zeros((3, N + 1))
    x_m = np.zeros((3, N + 1))
    x_m[:, 0] = np.squeeze(x0)
    P_p = np.zeros((3, 3, N + 1))
    R = np.identity(2) * 0.5

    for i in range(1, N + 1):
        if PART == "g":
            k = range(0, N, 3)
            if i in k:
                A = np.array([[1 - alpha_1, 0, 0], [0, 1 - alpha_2, 0], [alpha_1, alpha_2, 1 - alpha_3]])
            else:
                A = np.array([[1 - alpha_1, 0, 0], [0, 1, 0], [alpha_1, 0, 1 - alpha_3]])
        # prior update
        x_p[:, i] = A @ x_m[:, i - 1] + B @ u[:, i - 1]
        P_p[:, :, i] = A @ P[:, :, i - 1] @ np.transpose(A) + Q
        # measurement update
        P[:, :, i] = np.linalg.inv(np.linalg.inv(P_p[:, :, i]) + np.transpose(H) @ np.linalg.inv(R) @ H)
        x_m[:, i] = x_p[:, i] + P[:, :, i] @ np.transpose(H) @ np.linalg.inv(R) @ (z[:, i] - H @ x_p[:, i])

    # Plot results
    fig = plt.figure()
    subfigs = fig.subfigures(1, 2, wspace=0.07)
    # Mean
    axs_mean = subfigs[0].subplots(3, 1)
    for i, ax in enumerate(axs_mean):
        ax.plot(np.arange(0, N + 1), x[i, :], "k-", label="True state")
        ax.plot(
            np.arange(0, N + 1),
            x_m[i, :] + np.sqrt(P[i, i, :]),
            "r--",
            label="+/- 1 standard deviation",
        )
        ax.plot(np.arange(0, N + 1), x_m[i, :], "b--", label="Estimated state")
        ax.plot(np.arange(0, N + 1), x_m[i, :] - np.sqrt(P[i, i, :]), "r--")
        ax.set_xlabel("Time step k")
        ax.set_ylabel(f"Tank {i+1} Level, x({i+1})")
        ax.legend()

    # Variances
    ax_var = subfigs[1].subplots(1, 1)
    for row in range(3):
        for col in range(3):
            if col < row:
                continue
            subscript = str(row + 1) + str(col + 1)
            ax_var.plot(
                np.arange(0, N), P_p[row, col, 1:], label="$P_{" + subscript + "}$"
            )

    ax_var.set_xlabel("Time step k")
    ax_var.set_ylabel("Covariance matrix entry value")
    ax_var.legend()

    plt.show()