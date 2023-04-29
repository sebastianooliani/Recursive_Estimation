###########################################
# Realized by: Sebastiano Oliani
# 23/04/2023
###########################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy

if __name__ == "__main__":
    A = np.loadtxt("PS4/CubeModel_A.csv", delimiter=",")
    H = np.loadtxt("PS4/CubeModel_H.csv", delimiter=",")

    eig = np.linalg.eig(A)
    for i in range(len(eig)):
        if eig[0][i] >= 0:
            # print(eig[0][i])
            print("Unstable system!\n")
            break

    # Process noise variance: user input
    print("Choose process noise variance to run:")
    print(" (a) Q = 1e-6 I")
    print(" (b) Q = 1e-3 I")
    print(" (c) Q = 1e-9 I")
    print("Enter a, b, or c:")
    user_input = input()

    if user_input == "a":
        Q = 1e-06 * np.eye(16)
    elif user_input == "b":
        Q = 1e-03 * np.eye(16)
    elif user_input == "c":
        Q = 1e-09 * np.eye(16)
    else:
        raise ValueError('Invalid input. Please choose either "a" or "b" or "c".')

    # state variance
    S = np.identity(16) * 3 * 10**-4
    # gyro variance
    G = np.identity(2) * 2 * 10**-5 
    # encoder variance
    E = np.identity(1) * 10**-6
    V = scipy.linalg.block_diag(E, G, E, G, E, G, E, G, E, G, E, G)

    ## GENERATE DYNAMICS
    N = 100
    x = np.zeros((16, N + 1))
    x[:, 0] = np.squeeze(np.linalg.cholesky(S) @ np.random.randn(16))
    z = np.zeros((18, N + 1))
    u = np.zeros((6, N))
    v = np.linalg.cholesky(Q) @ np.random.randn(16, N)
    w = np.linalg.cholesky(V) @ np.random.randn(18, N + 1)

    for k in range(1, N+1):
        x[:, k] = A @ x[:, k-1] + v[:, k-1]
        z[:, k] = H @ x[:, k] + w[:, k]

    ## KALMAN FILTER IMPLEMENTATION
    ## time-varying Kalman filter
    x_p = np.zeros((16, N + 1))
    x_m = np.zeros((16, N + 1))
    x_m[:, 0] = np.squeeze(np.zeros((16, 1)))
    P_p = np.zeros((16, 16, N + 1))
    P_m = np.zeros((16, 16, N + 1))
    P_m[:, :, 0] = np.squeeze(S)

    for i in range(1, N + 1):
        # prior update
        x_p[:, i] = A @ x_m[:, i - 1]
        P_p[:, :, i] = A @ P_m[:, :, i - 1] @ A.T + Q
        # measurement update
        P_m[:, :, i] = np.linalg.inv(np.linalg.inv(P_p[:, :, i]) + H.T @ np.linalg.inv(V) @ H)
        x_m[:, i] = x_p[:, i] + P_m[:, :, i] @ H.T @ np.linalg.inv(V) @ (z[:, i] - H @ x_p[:, i])

    ## steady-state Kalman filter
    P_inf = scipy.linalg.solve_discrete_are(A.T, H.T, Q, V)
    K_inf = P_inf @ H.T @ np.linalg.inv(H @ P_inf @ H.T + V)
    x_ss = np.zeros((16, N + 1))
    x_ss[:, 0] = np.squeeze(np.zeros((16, 1)))

    for i in range(1, N + 1):
        x_ss[:, i] = (np.identity(16) - K_inf @ H) @ A @ x_ss[:, i - 1] + K_inf @ z[:, i]

    # Plots
    # Select what states to plot (select 4).
    sel = [1, 2, 13, 14]
    # Estimation error
    err1 = x - x_m
    err2 = x - x_ss

    fig = plt.figure()
    subfigs = fig.subfigures(1, 2, wspace=0.07)

    # Figure 1: states and state estimates
    axs_est = subfigs[0].subplots(4, 1)
    subfigs[0].suptitle("States and state estimates (in deg or deg/s)")
    for i, ax in enumerate(axs_est):
        ax.plot(
            np.arange(0, N + 1),
            x[sel[i] - 1, :] / np.pi * 180,
            label="true state",
            c="#2ca02c",
        )
        ax.plot(
            np.arange(0, N + 1),
            x_m[sel[i] - 1, :] / np.pi * 180,
            label="TVKF estimate",
            c="#1f77b4",
        )
        ax.plot(
            np.arange(0, N + 1),
            x_ss[sel[i] - 1, :] / np.pi * 180,
            label="SSKF estimate",
            c="#ff7f0e",
        )
        ax.set_ylabel(f"$x({sel[i]})$")
        ax.grid()
        if i == len(sel) - 1:
            ax.legend()
            ax.set_xlabel("Discrete-time step k")

    axs_err = subfigs[1].subplots(4, 1)
    subfigs[1].suptitle("Estimation error (in deg or deg/s)")
    for i, ax in enumerate(axs_err):
        ax.plot(
            np.arange(0, N + 1),
            err1[sel[i] - 1, :] / np.pi * 180,
            label="TVKF estimate",
            c="#1f77b4",
        )
        ax.plot(
            np.arange(0, N + 1),
            err2[sel[i] - 1, :] / np.pi * 180,
            label="SSKF estimate",
            c="#ff7f0e",
        )
        ax.set_ylabel(f"$x({sel[i]})$")
        ax.grid()
        if i == len(sel) - 1:
            ax.legend()
            ax.set_xlabel("Discrete-time step k")

    # Compute squared estimation error.
    print("\n\n# Squared estimation error:")
    print(f"Time-varying KF: {np.sum(err1**2)}")
    print(f"Steady-state KF: {np.sum(err2**2)}")

    plt.show()
            
        