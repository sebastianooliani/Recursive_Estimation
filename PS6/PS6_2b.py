###########################################
# Realized by: Sebastiano Oliani
# 24/06/2023
###########################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from util import place, obsvf

if __name__ == "__main__":
    N = 100 # time horizon

    A = np.array(
        [
            [6, 0.5, 1, 6.5],
            [5.25, -0.5, 1.25, 6.5],
            [-5, 0.5, 0, -5.5],
            [-1, -0.5, 0, -0.5]
        ]
    )
    
    B = np.array(
        [
            [0.5],
            [0.5],
            [0.5],
            [-0.5]
        ]
    )
    
    p = np.array(
        [0.8, 0.7] 
    )

    H = np.array(
        [
            [1, 0, 1, 2],
            [0, 0, 1, 1]
        ]
    )

    A, B, H, T = obsvf(A, B, H)
    K = place(A[2:, 2:].T, (H[0:, 2:] @ A[2:, 2:]).T, p).T
    K = np.vstack([np.zeros((2, 2)), K])
    K = T.T @ K
    print("\nGain matrix:", K)

    u = np.zeros((1, N))
    x = np.zeros((4, N))
    x_hat = np.zeros((4, N))

    # initial conditions
    x[:, 0] = [1, 1, 1, 1]
    x_hat[:, 0] = [0, 0, 0, 0]

    for i in range(1, N):
        x[:, i] = A @ x[:, i - 1] + B @ u[:, i - 1]
        x_hat[:, i] = A @ x_hat[:, i - 1] + B @ u[:, i - 1] + K @ (H @ x[:, i] - H @ (A @ x_hat[:, i - 1] - B @ u[:, i - 1]))

    e = x - x_hat

    fig, axs = plt.subplots(4)
    
    axs[0].plot(np.arange(0, N), e[0, :])
    axs[0].set_ylabel(f"Error state 1")

    axs[1].plot(np.arange(0, N), e[1, :])
    axs[1].set_ylabel(f"Error state 2")    

    axs[2].plot(np.arange(0, N), e[2, :])
    axs[2].set_ylabel(f"Error state 3")

    axs[3].plot(np.arange(0, N), e[3, :])
    axs[3].set_ylabel(f"Error state 4")
    
    fig.tight_layout()
    plt.show()