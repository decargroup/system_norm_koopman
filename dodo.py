import itertools
import pathlib
import pickle
from typing import Any, Dict, List
import shutil

import doit
import numpy as np
from matplotlib import pyplot as plt
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
# Path to ``datasets`` folder
DATASETS_DIR = WORKING_DIR.joinpath('datasets')
CONFIG_DIR = WORKING_DIR.joinpath('config')
LIFTING_FUNCTIONS_DIR = CONFIG_DIR.joinpath('lifting_functions')
REGRESSOR_DIR = CONFIG_DIR.joinpath('regressor')
EXPERIMENT = WORKING_DIR.joinpath('run_experiment.py')

HYDRA_PICKLE = 'run_experiment.pickle'

# log = 'run_experiment.log'
# PLOTS = [
#     # FASTER plots
#     ([
#         faster_eig,
#         faster_error,
#     ], [
#         f'faster__polynomial2__edmd/{log}',
#         f'faster__polynomial2__srconst_1/{log}',
#         f'faster__polynomial2__srconst_099/{log}',
#     ]),
#     # Soft robot EDMD plots
#     ([
#         soft_robot_error,
#         soft_robot_eig,
#         soft_robot_svd,
#         soft_robot_bode,
#         soft_robot_scatter_by_method,
#         soft_robot_weights,
#     ], [
#         f'soft_robot__polynomial3_delay1__edmd/{log}',
#         f'soft_robot__polynomial3_delay1__srconst_0999/{log}',
#         f'soft_robot__polynomial3_delay1__hinf/{log}',
#         f'soft_robot__polynomial3_delay1__hinfw/{log}',
#     ]),
#     # Soft robot DMDc plots
#     ([
#         soft_robot_dmdc_svd,
#         soft_robot_dmdc_bode,
#         soft_robot_scatter_dmdc,
#     ], [
#         f'soft_robot__polynomial3_delay1__srconst_0999/{log}',
#         f'soft_robot__polynomial3_delay1__srconst_0999_dmdc/{log}',
#         f'soft_robot__polynomial3_delay1__hinf/{log}',
#         f'soft_robot__polynomial3_delay1__hinf_dmdc/{log}',
#     ]),
#     # Soft robot performance plots
#     ([
#         soft_robot_exec,
#         soft_robot_ram,
#     ], [
#         'srconst_0999.dat',
#         'srconst_0999_dmdc.dat',
#         'hinf.dat',
#         'hinf_dmdc.dat',
#     ]),
# ]

# Okabe-Ito colorscheme
OKABE_ITO = {
    'black': (0.00, 0.00, 0.00),
    'orange': (0.90, 0.60, 0.00),
    'sky blue': (0.35, 0.70, 0.90),
    'bluish green': (0.00, 0.60, 0.50),
    'yellow': (0.95, 0.90, 0.25),
    'blue': (0.00, 0.45, 0.70),
    'vermillion': (0.80, 0.40, 0.00),
    'reddish purple': (0.80, 0.60, 0.70),
}
# Color mapping plots
C = {
    # Soft robot EDMD methods
    'edmd': OKABE_ITO['orange'],
    'srconst': OKABE_ITO['sky blue'],
    'hinf': OKABE_ITO['bluish green'],
    'hinfw': OKABE_ITO['reddish purple'],
    'hinfw_weight': OKABE_ITO['blue'],
    # Soft robot DMDc methods
    'srconst_dmdc': OKABE_ITO['vermillion'],
    'hinf_dmdc': OKABE_ITO['yellow'],
    # Soft robot inputs
    'u1': OKABE_ITO['vermillion'],
    'u2': OKABE_ITO['yellow'],
    'u3': OKABE_ITO['blue'],
    # FASTER EDMD methods
    '1.00': OKABE_ITO['orange'],
    '0.99': OKABE_ITO['sky blue'],
    # FASTER input
    'u': OKABE_ITO['bluish green'],
}
# Matplotlib settings
plt.rc('figure', dpi=100)
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=12)
plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')


# TODO add dependencies correctly?
# TODO Add clean for build too?
def task_build_dir() -> Dict[str, Any]:
    """Create ``build`` directory and subdirectories."""
    for (subdir_name, subdir) in BUILD_DIRS.items():
        yield {
            'name': subdir_name,
            'actions': [(doit.tools.create_folder, [subdir])],
            'targets': [subdir],
            'clean': [(shutil.rmtree, [subdir, True])]
        }


def task_pickle_faster_dataset() -> Dict[str, Any]:
    """Pickle FASTER dataset."""
    return {
        'actions': [pickle_faster_dataset],
        'file_dep': [DATASETS_DIR.joinpath('faster/faster.csv')],
        'targets': [BUILD_DIRS['datasets'].joinpath('faster.pickle')],
    }


def task_pickle_soft_robot_dataset() -> Dict[str, Any]:
    """Pickle soft robot dataset."""
    return {
        'actions': [pickle_soft_robot_dataset],
        'file_dep': [
            DATASETS_DIR.joinpath(
                'soft_robot/soft-robot-koopman/datafiles/softrobot_train-13_val-4.mat'
            )
        ],
        'targets': [BUILD_DIRS['datasets'].joinpath('soft_robot.pickle')],
    }


def task_experiment() -> Dict[str, Any]:
    """Run an experiment with Hydra.

    The possible experiments that can be run are decided by the top-level
    directories located in ``datasets/`` and the ``*.yaml`` files located in
    ``config/``. A task for every combination is created, but not all of them
    can be run by Hydra. For example, configuration files ending in ``*_base``
    can not be run, since they are sort of like "abstract classes".
    """
    # Figure out what dataset pickles **will be** created based on directories
    # located in ``datasets`` directory.
    datasets = [
        BUILD_DIRS['datasets'].joinpath(f'{ds.stem}.pickle')
        for ds in DATASETS_DIR.glob('*')
    ]
    # Find all options for lifting functions and regressors by looking at yamls
    lifting_functions = LIFTING_FUNCTIONS_DIR.glob('*.yaml')
    regressors = REGRESSOR_DIR.glob('*.yaml')
    # Compute every possible combination of dataset, lifting fn, and regressor
    experiments = itertools.product(datasets, lifting_functions, regressors)
    for (dataset, lifting_function, regressor) in experiments:
        # Form the name of the folder where the results will be stored
        exp_name = f'{dataset.stem}__{lifting_function.stem}__{regressor.stem}'
        # Form the complete path to the folder
        exp_dir = BUILD_DIRS['hydra_outputs'].joinpath(exp_name)
        yield {
            'name':
            exp_name,
            'actions': [
                f'python {EXPERIMENT} hydra.run.dir={exp_dir} '
                f'dataset={dataset} lifting_functions={lifting_function.stem} '
                f'regressor={regressor.stem}'
            ],
            'file_dep': [
                dataset,
                lifting_function,
                regressor,
            ],
            'targets': [exp_dir.joinpath(HYDRA_PICKLE)],
        }


def task_plot() -> Dict[str, Any]:
    """Plot a figure."""
    for action in [faster_error]:
        yield {
            'name':
            action.__name__,
            'actions': [action],
            'file_dep': [
                BUILD_DIRS['hydra_outputs'].joinpath(
                    'faster__polynomial2__edmd').joinpath(HYDRA_PICKLE),
                BUILD_DIRS['hydra_outputs'].joinpath(
                    'faster__polynomial2__srconst_099').joinpath(HYDRA_PICKLE),
                BUILD_DIRS['hydra_outputs'].joinpath(
                    'faster__polynomial2__srconst_1').joinpath(HYDRA_PICKLE),
            ],
            'targets': [
                BUILD_DIRS['figures'].joinpath(f'{action.__name__}.pdf'),
                BUILD_DIRS['cvd_figures'].joinpath(f'{action.__name__}.png'),
            ],
        }


def pickle_faster_dataset(dependencies: List[pathlib.Path],
                          targets: List[pathlib.Path]) -> None:
    """Create pickle of FASTER dataset."""
    array = np.loadtxt(dependencies[0], delimiter=',', skiprows=1).T
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
    with open(targets[0], 'wb') as f:
        pickle.dump(output_dict, f)


def pickle_soft_robot_dataset(dependencies: List[pathlib.Path],
                              targets: List[pathlib.Path]) -> None:
    """Create pickle of soft robot dataset."""
    # Load mat file
    mat = io.loadmat(dependencies[0], simplify_cells=True)
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
    with open(targets[0], 'wb') as f:
        pickle.dump(output_dict, f)


def faster_error(dependencies: List[pathlib.Path],
                 targets: List[pathlib.Path]) -> None:
    """Save faster timeseries plot."""
    unconst = _open_hydra_pickle(dependencies[0])
    const1 = _open_hydra_pickle(dependencies[1])
    const099 = _open_hydra_pickle(dependencies[2])

    t_step = 1 / unconst['bode']['f_samp']
    n_t = int(10 / t_step)
    t = np.arange(n_t) * t_step

    fig, ax = plt.subplots(3, 1, constrained_layout=True, sharex=True)
    ax[0].plot(
        t,
        unconst['timeseries_1.0']['X_validation'][:n_t, 1] -
        const1['timeseries_1.0']['X_prediction'][:n_t, 1],
        color=C['1.00'],
        label=r'A.S. constr., $\bar{\rho} = 1.00$',
    )
    ax[0].plot(
        t,
        unconst['timeseries_1.0']['X_validation'][:n_t, 1] -
        const099['timeseries_1.0']['X_prediction'][:n_t, 1],
        color=C['0.99'],
        label=r'A.S. constr., $\bar{\rho} = 0.99$',
    )

    ax[1].plot(
        t,
        unconst['timeseries_1.0']['X_validation'][:n_t, 2] -
        const1['timeseries_1.0']['X_prediction'][:n_t, 2],
        color=C['1.00'],
        label=r'A.S. constr., $\bar{\rho} = 1.00$',
    )
    ax[1].plot(
        t,
        unconst['timeseries_1.0']['X_validation'][:n_t, 2] -
        const099['timeseries_1.0']['X_prediction'][:n_t, 2],
        color=C['0.99'],
        label=r'A.S. constr., $\bar{\rho} = 0.99$',
    )

    ax[2].plot(
        t,
        unconst['timeseries_1.0']['X_validation'][:n_t, 3],
        '--',
        color=C['u'],
        label='Ground truth',
    )

    # ax[0].set_ylabel(r'$\Delta x_1(t)$ (N, norm.)')
    # ax[1].set_ylabel(r'$\Delta x_2(t)$ (m, norm.)')
    # ax[2].set_ylabel(r'$u(t)$ (V, norm.)')
    ax[0].set_ylabel(r'$\Delta x_1(t)$ (force)')
    ax[1].set_ylabel(r'$\Delta x_2(t)$ (deflection)')
    ax[2].set_ylabel(r'$u(t)$ (voltage)')
    ax[2].set_xlabel(r'$t$ (s)')

    ax[2].legend(
        ax[0].get_lines() + ax[2].get_lines(),
        [
            r'A.S. constr., $\bar{\rho} = 1.00$',
            r'A.S. constr., $\bar{\rho} = 0.99$',
            r'$u(t)$',
        ],
        loc='upper center',
        ncol=3,
        bbox_to_anchor=(0.5, -0.5),
    )

    ax[0].set_ylim(-1, 1)
    ax[1].set_ylim(-1, 1)
    ax[2].set_ylim(-1, 1)

    for target in targets:
        fig.savefig(target, bbox_inches='tight', pad_inches=0.1)


def _open_hydra_pickle(path: str) -> Dict[str, Any]:
    """Open pickles in directory of Hydra log and return dict of data."""
    with open(path, 'rb') as f:
        opened_pickle = pickle.load(f)
    return opened_pickle
