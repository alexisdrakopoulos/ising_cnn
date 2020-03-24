"""
script that uses metropolis_ising.py to generate systems,
can be multithreaded

This script is called using up to 2 arguments. The first argument is whether to
create new systems, to combine existing systmes, or to compress an existing large file.
The arguments are: "mcmc" or "combine" or "compress" respectively.
If "mcmc" is called, a second argument is required for number of steps, this must be int.

Examples:
    python3 generate_systems.py mcmc 5000
    python3 generate_systems.py combine
    python3 generate_systems.py compress

Contains:
    get_last_id - function to get last id assuming .npy files of systems
    create_large_np_array - combines .npy sys/label files into single arrays
    create_compressed_array - train/test split and combines sys/labels into compressed npz
    create_ising - calls mcmc_ising to create 1 ising system and saves local npy files
    run_multicore_mcmc - runs create_ising in multicore env using all available cores
"""

# system imports
from sys import argv
from configparser import ConfigParser, ExtendedInterpolation
from pathlib import Path
from glob import glob
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial

# math imports
import numpy as np
from numba import jit
from math import ceil

# our own module
from code.metropolis_ising import mcmc_ising

# Read config file for paths
config = ConfigParser(interpolation=ExtendedInterpolation())
config.read("code/config.ini")


def get_last_id():
    """
    Checks the last index (number of systems) that already exist

    Returns
        max_id - int of the number of systems in ising_systems/ dir
    """

    p = Path(f"{config['Paths']['ising data']}/")

    # Check if the folder exists
    if p.exists() is False:
        # if doesn't exist then no IDs to last id is 0
        return 0
    
    # get all .npy file name stems since they are the IDs
    all_files = [i.stem for i in p.rglob('*.npy')]

    max_id = 0
    for i in all_files:
        if i.isnumeric():
            if int(i) > max_id:
                max_id = int(i)

    return max_id


def create_large_np_array():
    """
    Creates a np.array of the Ising data, labels and IDs
    Assumes all Ising systems have same dimensionality n x n as 0.npy system
    Saves full_systems.npy and full_labels.npy to disk
    """

    max_id = get_last_id()

    # preallocate the labels and systems
    labels = np.zeros(max_id+1)
    n = np.load(f"{config['Paths']['ising data']}/0.npy").shape[0]
    systems = np.zeros([max_id+1, n, n])

    # now fill them in
    for i in tqdm(range(max_id+1)):
        systems[i] = np.load(f"{config['Paths']['ising data']}/" + str(i) + ".npy")
        labels[i] = np.load(f"{config['Paths']['ising data']}/" + str(i) + "_label.npy")

    print("Saving system and labels")
    np.save(f"{config['Paths']['ising data']}/full_systems.npy", systems)
    np.save(f"{config['Paths']['ising data']}/full_labels.npy", labels)


def create_compressed_array():
    """
    creates a train test split 80% train, 20% validation
    creates a compressed numpy array with keys training_data,
    training_labels, validaiton_data, validation_labels.
    """

    systems = np.load(f"{config['Paths']['ising data']}/full_systems.npy")
    labels = np.load(f"{config['Paths']['ising data']}/full_labels.npy")

    # choose 80% train, 20% validation
    idx_train = np.random.choice(list(range(labels.shape[0])),
                                 size=ceil(labels.shape[0]*0.8),
                                 replace=False)
    idx_val = list(set(list(range(labels.shape[0]))) - set(idx_train))

    # now save to ising_systems/ising_data.npz
    np.savez_compressed(f"{config['Paths']['ising data']}/ising_data.npz",
                        training_data = systems[idx_train],
                        training_labels = labels[idx_train],
                        validation_data = systems[idx_val],
                        validation_labels = systems[idx_val])


def create_ising(T, ID):
    """
    Generates 1 ising system through mcmc_ising, saves 2 np arrays, system and the temperature
    This function can be called alone but is usually called from run_multicore_mcmc
    
    Inputs:
        T - float (1 - 4 recommended) or "random_val" passed to mcmc_ising
        ID - int of ID, this is passed by run_multicore_mcmc
    Returns:
        ID - int of ID of system
        T - float of generated system label
        np.sum(lattice) - magnetization, purely for verbosity
    """

    p = Path(f"{config['Paths']['ising data']}/")

    # Check if the folder exists
    if p.exists() is False:
        p.mkdir(parents=True, exist_ok=True)

    sys_loc = p / str(ID)

    # run MCMC and save to disk
    lattice, T = mcmc_ising(T=T)

    # save lattice
    np.save(sys_loc, lattice)

    sys_loc = p / str(str(ID) + "_label")
    # save label (temperature)
    np.save(sys_loc, T)

    # return purely for print if verbose
    return ID, T, np.sum(lattice)


def run_multicore_mcmc(T, n_systems, verbose=True):
    """
    uses multiprocessing pool to call create_ising func
    to generate n_systems ising arrays using mcmc_ising func

    Inputs:
        T - float (1 - 4 recommended) or "random_val" str
        n_systems - int number of systems to generate
        verbose - Bool on whether to flush sys info after each sys is created
    """

    last_id = get_last_id()
    # if no systems exist we go back by 1, this fixed our range
    if last_id == 0:
        last_id = -1
    ids_to_generate = range(last_id + 1, last_id + n_systems + 1)

    with Pool() as pool:
        iterable = range(last_id + 1, last_id + 1 + n_systems)
        func = partial(create_ising, T)

        if verbose:
            for i in pool.imap_unordered(func, iterable):
                print(i)


if __name__ == '__main__':

    # first we parse the input arguments, this is a little messy
    if argv[1] == "mcmc":
        n_systems = int(argv[2])
        run_multicore_mcmc(T="random_val", n_systems=n_systems)

    elif argv[1] == "compress":
        create_compressed_array()

    elif argv[1] == "combine":
        create_large_np_array()

    else:
        raise ValueError("First arg must be 'mcmc', 'compress' or 'combine'")
    
    
    