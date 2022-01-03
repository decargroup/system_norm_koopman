"""Generate pickle file from source CSV file for `faster```."""

import argparse
import pathlib
import pickle

import numpy as np


def main():
    """Generate pickle file from source CSV file for `faster```."""
    # Set up arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('in_path')
    parser.add_argument('out_path')
    args = parser.parse_args()
    # Define source path
    in_path = pathlib.Path(args.in_path)
    # Define output path
    out_path = pathlib.Path(args.out_path)
    # Create pickle file
    create_pickle(in_path, out_path)


def create_pickle(in_path: pathlib.Path, out_path: pathlib.Path) -> None:
    """Generate pickle file from CSV file."""
    array = np.loadtxt(in_path, delimiter=',', skiprows=1).T
    t = array[0, :]
    r = array[1, :]
    u = array[2, :]
    y = array[3, :]
    d = array[4, :]
    # Get number of inputs
    n_u = 1
    # Get timestep
    t_step = np.mean(np.diff(t))
    # Compute episode feature
    val_set_len = t.size // 2
    train_set_len = t.size - val_set_len
    ep = np.concatenate((
        np.zeros((train_set_len, )),
        np.ones((val_set_len, )),
    ))
    # Form X
    X = np.vstack((
        ep,
        y / np.max(np.abs(y)),
        d / np.max(np.abs(d)),
        u / np.max(np.abs(u)),
    )).T
    # Create output dict
    output_dict = {
        'n_inputs': n_u,
        'episode_feature': True,
        't_step': t_step,
        'X': X,
        'training_episodes': [0],
        'validation_episodes': [1],
    }
    # Save pickle
    with open(out_path, 'wb') as f:
        pickle.dump(output_dict, f)


if __name__ == '__main__':
    main()
