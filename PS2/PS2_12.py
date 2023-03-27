###########################################
# Realized by: Sebastiano Oliani
# 26/03/2023
###########################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def initialize_dynamics(x, N, p):
    for i in range(1, N + 1):
        if np.random.rand() < p:
            x[i, 0] = np.mod(x[i - 1, 0] + 1, N)
        else:
            x[i, 0] = np.mod(x[i - 1, 0] - 1, N)
    return x

def measurement_update(zk, j, L, e):
    theta = 2 * np.pi * j / N
    if np.abs(zk - ((L - np.cos(theta)) ** 2 + np.sin(theta) ** 2) ** 0.5) < e:
        return 1 / (2 * e)
    else:
        return 0

if __name__ == "__main__":
    print('Choose part of the problem to run. Choose either "a" or "b": ')
    part = input()
    if part != "a" and part != "b":
        raise ValueError('Invalid input. Please choose either "a" or "b".')
    if part == "a":
        print('Choose sub-part of the problem to run. Press a number between 1 and 4: ')
        subpart = input()
        if subpart != "1" and subpart != "2" and subpart != "3" and subpart != "4":
            raise ValueError('Invalid input. Please choose a number between 1 and 4.')
    else:
        print('Choose sub-part of the problem to run. Press a number between 1 and 5: ')
        subpart = input()
        if subpart != "1" and subpart != "2" and subpart != "3" and subpart != "4" and subpart != "5":
            raise ValueError('Invalid input. Please choose a number between 1 and 5.')


    if part == "a":
        e = 0.5 # real params == estimate params
        if subpart == "1":
            L = 2
            p = 0.5
        elif subpart == "2":
            L = 2
            p = 0.55
        elif subpart == "3":
            L = 0.1
            p = 0.55
        elif subpart == "4":
            L = 0
            p = 0.55
        e_hat = e
        p_hat = p
    elif part == "b":
        L = 2
        e_hat = 0.5 # real parameters
        p_hat = 0.55
        if subpart == "1":
            e = 0.5 # estimates of the parameters
            p = 0.45
        elif subpart == "2":
            e = 0.5
            p = 0.5
        elif subpart == "3":
            e = 0.5
            p = 0.9
        elif subpart == "4":
            e = 0.9
            p = 0.55
        elif subpart == "5":
            e = 0.45
            p = 0.55
    
    print('\nSensor distance:', L)
    print('\nProbability of the noise associated to the model:', p_hat)
    print('\nProbability of the noise associated to the measurements:', e_hat, '\n')

    # enumerate state space X
    N = 100
    
    # posterior PDF: rows -> time index, columns -> location
    akk = np.zeros((N + 1, N))
    # prior PDF
    akk_1 = np.zeros((1, N))

    print("\nInitialization ...\n")
    # x0 is uniformly distributed on X = {0 ... N-1}
    akk[0, :] = 1 / N

    # dynamics
    x0 = N / 4
    x = np.zeros((N + 1, 1))
    x[0, 0] = x0
    x = initialize_dynamics(x, N, p_hat)
    # measurement vector
    z = np.zeros_like(x)

    print("\nRecursion ...\n")
    for i in range(1, N + 1):
        # prior update
        for j in range(N):
            akk_1[:, j] = p * akk[i - 1, np.mod(j - 1, N)] + (1 - p) * akk[i - 1, np.mod(j + 1, N)]
        
        # measurement
        theta = 2 * np.pi * x[i, 0] / N
        z[i] = ((L - np.cos(theta)) ** 2 + np.sin(theta) ** 2) ** 0.5 + e_hat * 2 * (np.random.rand() - 0.5)
        
        # measurement update
        for j in range(N):
            akk[i, j] = measurement_update(zk = z[i], j = j, L = L, e = e) * akk_1[0, j]

        # Normalize the weights of the posterior PDF: 
        # if the normalization is zero, it means that
        # we received an inconsistent measurement -> re-initialize the estimator.
        sum = np.sum(akk[i, :])

        if sum > 1e-06:
            akk[i, :] = akk[i, :] / sum
        else:
            akk[i, :] = akk[0, :]
            print(f"\nRe-initializing estimator at time step {i}!")
    
    print("\nEstimation ended!\n")

    print("\nVisualization\n")
    # Visualize the results
    fig = plt.figure()
    ax = plt.axes(projection = "3d")
    ax.set_xlabel("POSITION $x(k)/N$ ")
    ax.set_ylabel("TIME STEP $k$")
    X = np.arange(0, N) / N
    Y = np.arange(0, N + 1)
    X, Y = np.meshgrid(X, Y)
    ax.plot_surface(X, Y, akk, rstride = 1, cstride = 1, cmap = cm.coolwarm)
    ax.plot3D(
        x / N,
        np.arange(0, N + 1),
        np.ones((N + 1, 1)) * np.max(akk),
        label="Actual Position",
    )
    ax.legend()
    plt.show()