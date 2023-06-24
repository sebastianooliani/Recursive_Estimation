###########################################
# Realized by: Sebastiano Oliani
# 10/05/2023
###########################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from util import place

if __name__ == "__main__":
    N = 100 # time horizon

    A = np.array(
        [
            [0, -6],
            [1, 5]
        ]
    )
    
    B = np.array(
        [
            [1],
            [1]
        ]
    )
    
    p = np.array(
        [0.8, 0.7] 
    )

    H = np.array(
        [
            [0, 1]
        ]
    )

    K = place(A.T, (H @ A).T, p).T
    print("\nGain matrix:", K)

    u = np.zeros((1, N))
    x = np.zeros((2, N))
    x_hat = np.zeros((2, N))

    # initial conditions
    x[:, 0] = [1, 1]
    x_hat[:, 0] = [0, 0]

    for i in range(1, N):
        x[:, i] = A @ x[:, i - 1] + B @ u[:, i - 1]
        x_hat[:, i] = A @ x_hat[:, i - 1] + B @ u[:, i - 1] + K @ (H @ x[:, i] - H @ (A @ x_hat[:, i - 1] - B @ u[:, i - 1]))

    e = x - x_hat

    fig, axs = plt.subplots(4)
    
    axs[0].plot(np.arange(0, N), x[0, :], 'o-')
    axs[0].plot(np.arange(0, N), x_hat[0, :])
    axs[0].set_ylabel(f"State and estimate 1")

    axs[1].plot(np.arange(0, N), x[1, :], 'o-')
    axs[1].plot(np.arange(0, N), x_hat[1, :])
    axs[1].set_ylabel(f"State and estimate 2")    

    axs[2].plot(np.arange(0, N), e[0, :])
    axs[2].set_ylabel(f"Error state 1")

    axs[3].plot(np.arange(0, N), e[1, :])
    axs[3].set_ylabel(f"Error state 2")
    
    fig.tight_layout()
    plt.show()