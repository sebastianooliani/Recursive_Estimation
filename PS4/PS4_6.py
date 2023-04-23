###########################################
# Realized by: Sebastiano Oliani
# 23/04/2023
###########################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

if __name__ == "__main__":
    A = np.loadtxt("PS4/CubeModel_A.csv", delimiter=",")
    H = np.loadtxt("PS4/CubeModel_H.csv", delimiter=",")

    print(A)
    print(np.linalg.eig(A))
    eig = np.linalg.eig(A)
    for i in range(len(eig)):
        if eig[0][i] >= 0:
            print(eig[0][i])
            print("Unstable system!")
            
        