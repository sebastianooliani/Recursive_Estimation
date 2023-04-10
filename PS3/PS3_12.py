###########################################
# Realized by: Sebastiano Oliani
# 10/04/2023
###########################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

if __name__ == "__main__":
    print('Choose part of the problem to run. Choose either "b" or "c": ')
    part = input()
    N = 10000 # simulation length
    ########### TRUE VALUES GENERATION
    if part != "c" and part != "b":
        raise ValueError('Invalid input. Please choose either "b" or "c".')
    if part == "b":
        # normally distributed random variables
        x = np.random.randn()
    elif part == "c":
        # discrete random variables that take the values 1 and −1 with equal probability
        x = np.sign(np.random.randn())

    # Storage initialization
    x_est_lin = np.zeros((N + 1, 1))
    x_est_nl = np.zeros((N + 1, 1))

    # Simulation
    x_est_lin[0, 0] = 0
    x_est_nl[0, 0] = 0

    ################# RECURSIVE LEAST SQUARE ALGORITHM
    print('Start recursion')
    
    for k in range(1, N + 1):
        if part == "b":
            # normally distributed random variables
            w = np.random.randn()
        elif part == "c":
            # discrete random variables that take the values 1 and −1 with equal probability
            w = np.sign(np.random.randn())
        # Generate simulated measurement
        z = x + w

        # The linear estimator
        K = 1 / (1 + k)
        x_est_lin[k, 0] = x_est_lin[k - 1, 0] + K * (z - x_est_lin[k - 1, 0])
        #x_est_nl[k, 0] = x_est_nl[k - 1, 0] + K * (z - x_est_nl[k - 1, 0])

        # The nonlinear estimator, which only makes sense for discrete random variables.
        if np.abs(x_est_nl[k-1, 0]) != 0:
            # If we have already determined what x is
            x_est_nl[k, 0] = x_est_nl[k-1, 0]
        else:
            # Otherwise, we have to look at the measurement.  If z = 0, then it
            # is equally likely that x = 1 or -1.  If z = 2, then x must be 1.
            # If z = -2, then x must be -1
            if np.abs(z) < 1:
                x_est_nl[k, 0] = 0
            else:
                if z > 1:
                    x_est_nl[k, 0] = 1
                else:
                    x_est_nl[k, 0] = -1

    print('End Recursion')

    # Plot the results.
    fig = plt.figure()
    ax = plt.axes()
    if part == "b":
        ax.plot(np.arange(0, N + 1), x_est_lin, "*", label="Linear Estimate", alpha=0.2)
        ax.plot(
            np.arange(0, N + 1),
            np.ones(N + 1) * x,
            "r",
            label="Actual",
        )
    else:
        # The nonlinear estimator only makes sense for the DRV case.
        ax.plot(np.arange(0, N + 1), x_est_lin, "*", label="Linear Estimate", alpha=0.2)
        ax.plot(
            np.arange(0, N + 1),
            x_est_nl,
            "x",
            label="Non-linear Estimate",
            alpha=0.2,
        )
        ax.plot(
            np.arange(0, N + 1),
            np.ones(N + 1) * x,
            "r",
            label="Actual",
        )
    ax.legend()
    plt.show()