import numpy as np
from numba import jit
import os

def mcmc_ising(n=128,
               nsteps=500_000_000,
               T=1,
               J=1,
               H=0):
    """
    This function performs Metropolis MCMC algorithm on a 2D Ising lattice of
    size nxn using numba to JIT compile. It can do around 25,000,000+ iters/second.
    It will return the 2D numpy array as well as the steps and the temperature used.

    Inputs:
        n - int specifying width and height of system n x n
        nsteps - int specifying number of metropolis iterations
        T - float (1 - 4 recommended) or str "random_val" for random unif (1, 4)
        J - float (1 recommended) of inter-cell strength
        H - float (0 recommended) of magnetism
    Returns:
        ising_calculation(ising_lattice) - n x n np array of ising system
        T - float entered or generated from random.uniform
    """

    # initialize our lattice
    ising_lattice = np.random.choice([1, -1], size=(n, n))
    ising_lattice = ising_lattice.astype(np.int8)

    # some errors with generating random values in multiprocessing
    # so this is here to fix that
    if T == "random_val":
        np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
        T = round(np.random.uniform(1, 4), 3)

    # From here on down we are in numba jit
    @jit(nopython=True)
    def ising_calculation(lattice):
        # Pre-calculate the temperature
        pre_temp = 1 / T

        for step in range(nsteps):

            # Randomly choose i j to index into array
            i = np.random.randint(n)
            j = np.random.randint(n)

            # Boundary conditions and neighbours
            Sn = lattice[(i - 1) % n, j] + lattice[(i + 1) % n, j] + \
                 lattice[i, (j - 1) % n] + lattice[i, (j + 1) % n]

            # Calculating local change in energy
            dE = 2 * lattice[i, j] * (H + J * Sn)

            # Metropolis check
            if np.random.random() < np.exp(-dE * pre_temp):
                lattice[i, j] = -lattice[i, j]

        return lattice

    return ising_calculation(ising_lattice), T