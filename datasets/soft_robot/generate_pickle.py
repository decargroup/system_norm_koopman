"""Generate pickle file from source MATLAB file for ``soft_robot``."""

import argparse
import pathlib
import pickle

import numpy as np
from scipy import io  # type: ignore


def main():
    """Generate pickle file from source MATLAB file for ``soft_robot``."""
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
    """Generate pickle file from mat file."""
    # Load mat file
    mat = io.loadmat(in_path, simplify_cells=True)
    # Get number of inputs
    n_u = mat['train'][0]['u'].shape[1]
    # Get number of training and validation episodes
    n_train = len(mat['train'])
    n_val = len(mat['val'])
    # Get timestep
    t_step = np.mean(np.diff(mat['train'][0]['t']))
    # Form data matrix
    X_lst = []
    train_ep = []
    ep_idx = 0
    for i in range(n_train):
        y = mat['train'][i]['y'] * 2.54  # ``in`` to ``cm``
        u = mat['train'][i]['u']
        e = ep_idx * np.ones((y.shape[0], 1))
        x = np.hstack((e, y, u))
        X_lst.append(x)
        train_ep.append(ep_idx)
        ep_idx += 1
    val_ep = []
    for i in range(n_val):
        y = mat['val'][i]['y'] * 2.54  # ``in`` to ``cm``
        u = mat['val'][i]['u']
        e = ep_idx * np.ones((y.shape[0], 1))
        x = np.hstack((e, y, u))
        if i == 2:
            X_lst.append(x[100:, :])
        else:
            X_lst.append(x)
        val_ep.append(ep_idx)
        ep_idx += 1
    X = np.vstack(X_lst)
    # Create output dict
    output_dict = {
        'n_inputs': n_u,
        'episode_feature': True,
        't_step': t_step,
        'X': X,
        'training_episodes': train_ep,
        'validation_episodes': val_ep,
    }
    # Save pickle
    with open(out_path, 'wb') as f:
        pickle.dump(output_dict, f)


if __name__ == '__main__':
    main()
