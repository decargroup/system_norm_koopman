import pathlib
import pickle
from typing import Any, Dict

import numpy as np
from scipy import io

# DOIT_CONFIG = {'default_tasks': []}

# Directory containing ``dodo.py``
WORKING_DIR = pathlib.Path(__file__).parent.resolve()
# Path to ``build`` folder
BUILD_DIR = WORKING_DIR.joinpath('build')
# Dict of subfolders in ``build``
BUILD_DIRS = {
    dir: BUILD_DIR.joinpath(dir)
    for dir in [
        'datasets',
        'figures',
        'hydra_outputs',
        'mprof_outputs',
        'cvd_figures',
    ]
}
# Path ro ``datasets`` folder
DATASETS_DIR = WORKING_DIR.joinpath('datasets')


def task_build_dir() -> Dict[str, Any]:
    """Create ``build`` directory and subdirectories."""
    def make_subdir(subdir: pathlib.Path) -> None:
        subdir.mkdir(parents=True, exist_ok=True)

    for (subdir_name, subdir) in BUILD_DIRS.items():
        yield {
            'name': subdir_name,
            'actions': [(make_subdir, [subdir])],
            'targets': [subdir],
        }


def task_pickle_faster_dataset() -> Dict[str, Any]:
    """Create pickle of FASTER dataset."""
    def create_pickle(in_path: pathlib.Path, out_path: pathlib.Path) -> None:
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

    in_path = DATASETS_DIR.joinpath('faster/faster.csv')
    out_path = BUILD_DIRS['datasets'].joinpath('faster.pickle')

    return {
        'actions': [(create_pickle, [in_path, out_path])],
        'file_dep': [in_path],
        'targets': [out_path],
    }


def task_pickle_soft_robot_dataset() -> Dict[str, Any]:
    """Create pickle of soft robot dataset."""
    def create_pickle(in_path: pathlib.Path, out_path: pathlib.Path) -> None:
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

    in_path = DATASETS_DIR.joinpath(
        'soft_robot/soft-robot-koopman/datafiles/softrobot_train-13_val-4.mat')
    out_path = BUILD_DIRS['datasets'].joinpath('soft_robot.pickle')

    return {
        'actions': [(create_pickle, [in_path, out_path])],
        'file_dep': [in_path],
        'targets': [out_path],
    }
