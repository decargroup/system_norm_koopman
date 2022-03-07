"""Define and automate tasks with ``doit``.

This file is ``doit``'s equivalent of a ``Makefile``. When you run any ``doit``
command, all the task definition functions (functions starting with ``task_*``)
are run. These functions determine which files tasks produce, which files they
require, and what actions need to be taken to generate those files. To see a
list of all available tasks, run::

    $ doit list --all

If you have a powerful enough computer and want to generate all the plots from
scratch, you can run::

    $ doit

This will run all the ``plot:*`` tasks, along with all the ``experiment:*`` and
``profile:*`` tasks they depend on. This can take more than 8 hours and 16 GiB
of RAM. A more reasonable task for a laptop is::

    $ doit plot:faster*

If you built ``./build/hydra_outputs/`` on one machine and want to adjust the
plots on another machine, you will need to run::

    $ doit reset-dep

This will ensure that ``doit`` will recognize the files you just moved as
up-to-date.

For more information, check out https://pydoit.org/
"""

import itertools
import pathlib
import pickle
import re
import shutil
from typing import Any, Dict, Generator, List, Tuple

import cmcrameri
import doit
import matplotlib
import numpy as np
import pandas
from matplotlib import pyplot as plt
from scipy import io, linalg

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# Configure ``doit`` to run plot tasks by default
DOIT_CONFIG = {'default_tasks': ['plot']}

# Directory containing ``dodo.py``
WORKING_DIR = pathlib.Path(__file__).parent.resolve()
# Path to ``build/`` folder
BUILD_DIR = WORKING_DIR.joinpath('build')
# Dict of subfolders in ``build/``
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
# Path to ``datasets/`` folder
DATASETS_DIR = WORKING_DIR.joinpath('datasets')
# Path to ``config/`` folder
CONFIG_DIR = WORKING_DIR.joinpath('config')
# Dict of subfolders in ``config/``
CONFIG_DIRS = {
    dir: CONFIG_DIR.joinpath(dir)
    for dir in [
        'lifting_functions',
        'regressor',
    ]
}
# Path to ``run_experiment.py`` script
EXPERIMENT_PY = WORKING_DIR.joinpath('run_experiment.py')
# Name of data pickle within ``build/hydra_outputs/*/`` directories
HYDRA_PICKLE = 'run_experiment.pickle'

# H-infinity LaTeX
HINF = r'$\mathcal{H}_\infty$'
# Okabe-Ito colorscheme: https://jfly.uni-koeln.de/color/
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
# Color mapping for plots
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
    # Tikz
    'tikz_x1': OKABE_ITO['orange'],
    'tikz_x2': OKABE_ITO['sky blue'],
    'tikz_u': OKABE_ITO['bluish green'],
    'tikz_rho': OKABE_ITO['blue'],
    'tikz_hinf': OKABE_ITO['vermillion'],
    # 'tikz_eig': OKABE_ITO['black'],
    # 'tikz_bode': OKABE_ITO['black'],
    'tikz_eig': cmcrameri.cm.batlow(0),
    'tikz_bode': cmcrameri.cm.batlow(0),
}
# Global Matplotlib settings
if matplotlib.checkdep_usetex(True):  # Use LaTeX only if available
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=16)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')
# Figure saving options
SAVEFIG_PARAMS: Dict[str, Any] = {
    'bbox_inches': 'tight',
    'pad_inches': 0.1,
}
SAVEFIG_TIKZ_PARAMS: Dict[str, Any] = {
    'bbox_inches': 'tight',
    'pad_inches': 0.01,
}

# --------------------------------------------------------------------------- #
# Task definitions
# --------------------------------------------------------------------------- #


def task_directory() -> Generator[Dict[str, Any], None, None]:
    """Create ``build/`` directory and subdirectories."""
    # Create parent directory
    yield {
        'name': BUILD_DIR.stem,
        'actions': [(doit.tools.create_folder, [BUILD_DIR])],
        'targets': [BUILD_DIR],
        'clean': [(shutil.rmtree, [BUILD_DIR, True])],
        'uptodate': [True],
    }
    # Create subdirectories
    for subdir in BUILD_DIRS.values():
        yield {
            'name': BUILD_DIR.stem + '/' + subdir.stem,
            'actions': [(doit.tools.create_folder, [subdir])],
            'task_dep': [f'directory:{BUILD_DIR.stem}'],
            'targets': [subdir],
            'clean': [(shutil.rmtree, [subdir, True])],
            'uptodate': [True],
        }


def task_pickle() -> Generator[Dict[str, Any], None, None]:
    """Pickle a dataset for use with Hydra."""
    # FASTER dataset
    yield {
        'name': 'faster',
        'actions': [pickle_faster_dataset],
        'file_dep': [DATASETS_DIR.joinpath('faster/faster.csv')],
        'task_dep': ['directory:build/datasets'],
        'targets': [BUILD_DIRS['datasets'].joinpath('faster.pickle')],
        'clean': True,
    }
    # Soft robot dataset
    soft_robot_file = DATASETS_DIR.joinpath(
        'soft_robot/soft-robot-koopman/datafiles/softrobot_train-13_val-4.mat')
    yield {
        'name': 'soft_robot',
        'actions': [pickle_soft_robot_dataset],
        'file_dep': [soft_robot_file],
        'task_dep': ['directory:build/datasets'],
        'targets': [BUILD_DIRS['datasets'].joinpath('soft_robot.pickle')],
        'clean': True,
    }


def task_experiment() -> Generator[Dict[str, Any], None, None]:
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
    lifting_functions = CONFIG_DIRS['lifting_functions'].glob('*.yaml')
    regressors = CONFIG_DIRS['regressor'].glob('*.yaml')
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
                f'python {EXPERIMENT_PY} hydra.run.dir={exp_dir} '
                f'dataset={dataset} lifting_functions={lifting_function.stem} '
                f'regressor={regressor.stem}'
            ],
            'file_dep': [dataset, lifting_function, regressor],
            'task_dep': ['directory:build/hydra_outputs'],
            'targets': [exp_dir.joinpath(HYDRA_PICKLE)],
            'clean': [(shutil.rmtree, [exp_dir, True])],
        }


def task_profile() -> Generator[Dict[str, Any], None, None]:
    """Profile an experiment with Memory Profiler."""
    dataset = BUILD_DIRS['datasets'].joinpath('soft_robot.pickle')
    lifting_function = CONFIG_DIRS['lifting_functions'].joinpath(
        'polynomial3_delay1.yaml')
    regressors = [
        CONFIG_DIRS['regressor'].joinpath('srconst_0999.yaml'),
        CONFIG_DIRS['regressor'].joinpath('srconst_0999_dmdc.yaml'),
        CONFIG_DIRS['regressor'].joinpath('hinf.yaml'),
        CONFIG_DIRS['regressor'].joinpath('hinf_dmdc.yaml'),
    ]
    for regressor in regressors:
        exp_name = (f'{dataset.stem}__{lifting_function.stem}'
                    f'__{regressor.stem}__max_iter_1')
        exp_dir = BUILD_DIRS['mprof_outputs'].joinpath(exp_name)
        prof_dir = BUILD_DIRS['mprof_outputs'].joinpath(
            f'{regressor.stem}.dat')
        yield {
            'name':
            regressor.stem,
            'actions': [
                f'mprof run --include-children --output {prof_dir} '
                f'--python {EXPERIMENT_PY} '
                f'dataset={dataset} lifting_functions={lifting_function.stem} '
                f'regressor={regressor.stem} regressor.regressor.max_iter=1 '
                f'profile=True hydra.run.dir={exp_dir}'
            ],
            'file_dep': [dataset, lifting_function, regressor],
            'task_dep': ['directory:build/mprof_outputs'],
            'targets': [prof_dir, exp_dir.joinpath(HYDRA_PICKLE)],
            'clean':
            [doit.task.clean_targets, (shutil.rmtree, [exp_dir, True])],
        }


def task_plot() -> Generator[Dict[str, Any], None, None]:
    """Plot a figure."""
    for action in [
            faster_eig,
            faster_error,
            faster_tikz_time_1a,
            faster_tikz_time_1b,
            faster_tikz_time_1c,
            faster_tikz_time_2a,
            faster_tikz_time_2b,
            faster_tikz_time_2c,
            faster_tikz_time_3a,
            faster_tikz_time_3b,
            faster_tikz_time_3c,
            faster_tikz_lf_1,
            faster_tikz_lf_2,
            faster_tikz_lf_3,
            faster_tikz_lifted_1a,
            faster_tikz_lifted_1b,
            faster_tikz_lifted_1c,
            faster_tikz_lifted_2a,
            faster_tikz_lifted_2b,
            faster_tikz_lifted_2c,
            faster_tikz_lifted_3a,
            faster_tikz_lifted_3b,
            faster_tikz_lifted_3c,
            faster_tikz_eig,
            faster_tikz_bode,
    ]:
        yield {
            'name':
            action.__name__,
            'actions': [action],
            'file_dep': [
                BUILD_DIRS['hydra_outputs'].joinpath(
                    'faster__polynomial2__edmd').joinpath(HYDRA_PICKLE),
                BUILD_DIRS['hydra_outputs'].joinpath(
                    'faster__polynomial2__srconst_1').joinpath(HYDRA_PICKLE),
                BUILD_DIRS['hydra_outputs'].joinpath(
                    'faster__polynomial2__srconst_099').joinpath(HYDRA_PICKLE),
            ],
            'task_dep': ['directory:build/figures'],
            'targets': [
                BUILD_DIRS['figures'].joinpath(f'{action.__name__}.pdf'),
                BUILD_DIRS['figures'].joinpath(f'{action.__name__}.png'),
            ],
            'clean':
            True,
            'uptodate': [False],
        }
    for action in [
            soft_robot_error,
            soft_robot_eig,
            soft_robot_bode,
            soft_robot_svd,
            soft_robot_weights,
            soft_robot_scatter_by_method,
    ]:
        yield {
            'name':
            action.__name__,
            'actions': [action],
            'file_dep': [
                BUILD_DIRS['hydra_outputs'].joinpath(
                    'soft_robot__polynomial3_delay1__edmd').joinpath(
                        HYDRA_PICKLE),
                BUILD_DIRS['hydra_outputs'].joinpath(
                    'soft_robot__polynomial3_delay1__srconst_0999').joinpath(
                        HYDRA_PICKLE),
                BUILD_DIRS['hydra_outputs'].joinpath(
                    'soft_robot__polynomial3_delay1__hinf').joinpath(
                        HYDRA_PICKLE),
                BUILD_DIRS['hydra_outputs'].joinpath(
                    'soft_robot__polynomial3_delay1__hinfw').joinpath(
                        HYDRA_PICKLE),
            ],
            'task_dep': ['directory:build/figures'],
            'targets': [
                BUILD_DIRS['figures'].joinpath(f'{action.__name__}.pdf'),
                BUILD_DIRS['figures'].joinpath(f'{action.__name__}.png'),
            ],
            'clean':
            True,
            'uptodate': [False],
        }
    for action in [
            soft_robot_dmdc_svd,
            soft_robot_dmdc_bode,
            soft_robot_scatter_dmdc,
    ]:
        yield {
            'name':
            action.__name__,
            'actions': [action],
            'file_dep': [
                BUILD_DIRS['hydra_outputs'].joinpath(
                    'soft_robot__polynomial3_delay1__srconst_0999').joinpath(
                        HYDRA_PICKLE),
                BUILD_DIRS['hydra_outputs'].joinpath(
                    'soft_robot__polynomial3_delay1__srconst_0999_dmdc').
                joinpath(HYDRA_PICKLE),
                BUILD_DIRS['hydra_outputs'].joinpath(
                    'soft_robot__polynomial3_delay1__hinf').joinpath(
                        HYDRA_PICKLE),
                BUILD_DIRS['hydra_outputs'].joinpath(
                    'soft_robot__polynomial3_delay1__hinf_dmdc').joinpath(
                        HYDRA_PICKLE),
            ],
            'task_dep': ['directory:build/figures'],
            'targets': [
                BUILD_DIRS['figures'].joinpath(f'{action.__name__}.pdf'),
                BUILD_DIRS['figures'].joinpath(f'{action.__name__}.png'),
            ],
            'clean':
            True,
            'uptodate': [False],
        }
    for action in [soft_robot_ram, soft_robot_exec]:
        yield {
            'name':
            action.__name__,
            'actions': [action],
            'file_dep': [
                BUILD_DIRS['mprof_outputs'].joinpath('srconst_0999.dat'),
                BUILD_DIRS['mprof_outputs'].joinpath('srconst_0999_dmdc.dat'),
                BUILD_DIRS['mprof_outputs'].joinpath('hinf.dat'),
                BUILD_DIRS['mprof_outputs'].joinpath('hinf_dmdc.dat'),
            ],
            'task_dep': ['directory:build/figures'],
            'targets': [
                BUILD_DIRS['figures'].joinpath(f'{action.__name__}.pdf'),
                BUILD_DIRS['figures'].joinpath(f'{action.__name__}.png'),
            ],
            'clean':
            True,
            'uptodate': [False],
        }


@doit.create_after(
    executed='plot',
    target_regex=rf'{BUILD_DIRS["cvd_figures"].resolve()}/.*\.png',
)
def task_cvd() -> Generator[Dict[str, Any], None, None]:
    """Simulate color vision deficiency a plot."""
    plots = BUILD_DIRS['figures'].glob('*.png')
    methods = ['protan', 'deutan', 'tritan']
    tasks = itertools.product(plots, methods)
    for (plot, method) in tasks:
        file_dep = BUILD_DIRS['figures'].joinpath(plot)
        target = BUILD_DIRS['cvd_figures'].joinpath(
            f'{plot.stem}_{method}.png')
        yield {
            'name': f'{plot.stem}_{method}',
            'actions': [f'daltonlens-python -d {method} {file_dep} {target}'],
            'file_dep': [file_dep],
            'task_dep': ['directory:build/cvd_figures'],
            'targets': [target],
            'clean': True,
            'uptodate': [False],
        }


# --------------------------------------------------------------------------- #
# Task actions
# --------------------------------------------------------------------------- #


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
    deps = _open_hydra_pickles(dependencies)
    unconst = deps['faster__polynomial2__edmd']
    const1 = deps['faster__polynomial2__srconst_1']
    const099 = deps['faster__polynomial2__srconst_099']
    # Compute time array
    t_step = 1 / unconst['bode']['f_samp']
    n_t = int(10 / t_step)
    t = np.arange(n_t) * t_step
    # Create figure
    fig, ax = plt.subplots(
        3,
        1,
        constrained_layout=True,
        sharex=True,
        figsize=(5, 5),
    )
    # Plot first state
    ax[0].plot(
        t,
        unconst['timeseries_1.0']['X_validation'][:n_t, 1]
        - const1['timeseries_1.0']['X_prediction'][:n_t, 1],
        color=C['1.00'],
        label=r'A.S. constr., $\bar{\rho} = 1.00$',
    )
    ax[0].plot(
        t,
        unconst['timeseries_1.0']['X_validation'][:n_t, 1]
        - const099['timeseries_1.0']['X_prediction'][:n_t, 1],
        color=C['0.99'],
        label=r'A.S. constr., $\bar{\rho} = 0.99$',
    )
    # Plot second state
    ax[1].plot(
        t,
        unconst['timeseries_1.0']['X_validation'][:n_t, 2]
        - const1['timeseries_1.0']['X_prediction'][:n_t, 2],
        color=C['1.00'],
        label=r'A.S. constr., $\bar{\rho} = 1.00$',
    )
    ax[1].plot(
        t,
        unconst['timeseries_1.0']['X_validation'][:n_t, 2]
        - const099['timeseries_1.0']['X_prediction'][:n_t, 2],
        color=C['0.99'],
        label=r'A.S. constr., $\bar{\rho} = 0.99$',
    )
    # Plot input
    ax[2].plot(
        t,
        unconst['timeseries_1.0']['X_validation'][:n_t, 3],
        '--',
        color=C['u'],
        label='Ground truth',
    )
    # Set labels
    ax[0].set_ylabel(r'$\Delta x_1(t)$'
                     '\n(force)')
    ax[1].set_ylabel(r'$\Delta x_2(t)$'
                     '\n(deflection)')
    ax[2].set_ylabel(r'$u(t)$'
                     '\n(voltage)')
    ax[2].set_xlabel(r'$t$ (s)')
    # Create legend
    fig.legend(
        ax[0].get_lines() + ax[2].get_lines(),
        [
            r'A.S. constr., $\bar{\rho} = 1.00$',
            r'A.S. constr., $\bar{\rho} = 0.99$',
            r'$u(t)$',
        ],
        loc='upper center',
        ncol=3,
        bbox_to_anchor=(0.5, 0),
    )
    # Set axis limits
    ax[0].set_ylim(-1, 1)
    ax[1].set_ylim(-1, 1)
    ax[2].set_ylim(-1, 1)
    ax[0].set_yticks([-1, -0.5, 0, 0.5, 1])
    ax[1].set_yticks([-1, -0.5, 0, 0.5, 1])
    ax[2].set_yticks([-1, -0.5, 0, 0.5, 1])
    # Save targets
    for target in targets:
        fig.savefig(target, **SAVEFIG_PARAMS)


def faster_eig(dependencies: List[pathlib.Path],
               targets: List[pathlib.Path]) -> None:
    """Save FASTER eigenvalue plot."""
    deps = _open_hydra_pickles(dependencies)
    unconst = deps['faster__polynomial2__edmd']
    const1 = deps['faster__polynomial2__srconst_1']
    const099 = deps['faster__polynomial2__srconst_099']
    # Create figure
    fig = plt.figure(constrained_layout=True, figsize=(5, 5))
    ax = fig.add_subplot(projection='polar')
    # Set common scatter plot settings
    style = {
        's': 50 * 1.5,
        'edgecolors': 'w',
        'linewidth': 0.25 * 1.5,
    }
    # Plot eigenvalue constraints
    th = np.linspace(0, 2 * np.pi)
    ax.plot(th, np.ones(th.shape), '--', color=C['1.00'], linewidth=1.5)
    ax.plot(th, 0.99 * np.ones(th.shape), '--', color=C['0.99'], linewidth=1.5)
    # Plot eigenvalues
    ax.scatter(
        np.angle(const1['eigenvalues']['eigv']),
        np.absolute(const1['eigenvalues']['eigv']),
        color=C['1.00'],
        marker='o',
        label=r'A.S. constr., $\bar{\rho} = 1.00$',
        **style,
    )
    ax.scatter(
        np.angle(const099['eigenvalues']['eigv']),
        np.absolute(const099['eigenvalues']['eigv']),
        color=C['0.99'],
        marker='s',
        label=r'A.S. constr., $\bar{\rho} = 0.99$',
        **style,
    )
    # Add axis labels
    ax.text(0, 1.125, r'$\angle \lambda_i$')
    ax.text(-np.pi / 8 - np.pi / 16, 0.5, r'$|\lambda_i|$')
    ax.set_axisbelow(True)
    # Create legend
    ax.legend(loc='lower left', ncol=1)
    # Set axis limits and ticks
    ax.set_xticks([d * np.pi / 180 for d in [-20, -10, 0, 10, 20]])
    ax.set_thetalim(-np.pi / 8, np.pi / 8)
    # Save targets
    for target in targets:
        fig.savefig(target, **SAVEFIG_PARAMS)


def soft_robot_error(dependencies: List[pathlib.Path],
                     targets: List[pathlib.Path]) -> None:
    """Save soft robot timeseries plot."""
    deps = _open_hydra_pickles(dependencies)
    edmd = deps['soft_robot__polynomial3_delay1__edmd']
    srconst = deps['soft_robot__polynomial3_delay1__srconst_0999']
    hinf = deps['soft_robot__polynomial3_delay1__hinf']
    hinfw = deps['soft_robot__polynomial3_delay1__hinfw']
    # Select timeseries to plot
    series = 'timeseries_15.0'
    # Calculate time arrays
    t_step = 1 / edmd['bode']['f_samp']
    n_t = edmd[series]['X_validation'].shape[0]
    t = np.arange(n_t) * t_step
    # Get state dimension
    n_x = edmd[series]['X_prediction'].shape[1] - 1
    # Create figure
    fig, ax = plt.subplots(
        3,
        1,
        constrained_layout=True,
        sharex=True,
        figsize=(5, 5),
    )
    # Plot errors
    for i in range(2):
        ax[i].plot(
            t,
            (edmd[series]['X_validation'][:n_t, i + 1]
             - edmd[series]['X_prediction'][:n_t, i + 1]),
            label='Extended DMD',
            color=C['edmd'],
        )
        ax[i].plot(
            t,
            (edmd[series]['X_validation'][:n_t, i + 1]
             - srconst[series]['X_prediction'][:n_t, i + 1]),
            label='A.S. constraint',
            color=C['srconst'],
        )
        ax[i].plot(
            t,
            (edmd[series]['X_validation'][:n_t, i + 1]
             - hinf[series]['X_prediction'][:n_t, i + 1]),
            label=f'{HINF} regularizer',
            color=C['hinf'],
        )
    # Plot inputs
    ax[2].plot(
        t,
        edmd[series]['X_validation'][:n_t, 3],
        '--',
        color=C['u1'],
        label=r'$u_1(t)$',
    )
    ax[2].plot(
        t,
        edmd[series]['X_validation'][:n_t, 4],
        '--',
        color=C['u2'],
        label=r'$u_2(t)$',
    )
    ax[2].plot(
        t,
        edmd[series]['X_validation'][:n_t, 5],
        '--',
        color=C['u3'],
        label=r'$u_3(t)$',
    )
    # Set axis labels
    ax[-1].set_xlabel(r'$t$ (s)')
    ax[0].set_ylabel(r'$\Delta x_1(t)$ (cm)')
    ax[1].set_ylabel(r'$\Delta x_2(t)$ (cm)')
    ax[2].set_ylabel(r'${\bf u}(t)$ (V)')
    # Set axis limits
    ax[0].set_ylim(-5, 5)
    ax[1].set_ylim(-5, 5)
    ax[2].set_ylim(-1, 9)
    # Set axis ticks
    ax[0].set_yticks([-4, -2, 0, 2, 4])
    ax[1].set_yticks([-4, -2, 0, 2, 4])
    ax[2].set_yticks([0, 2, 4, 6, 8])
    # Create legend
    fig.legend(
        [
            ax[1].get_lines()[0],
            ax[2].get_lines()[0],
            ax[1].get_lines()[1],
            ax[2].get_lines()[1],
            ax[1].get_lines()[2],
            ax[2].get_lines()[2],
        ],
        [
            'Extended DMD',
            r'$u_1(t)$',
            'A.S. constraint',
            r'$u_2(t)$',
            f'{HINF} regularizer',
            r'$u_3(t)$',
        ],
        loc='upper center',
        ncol=3,
        bbox_to_anchor=(0.5, 0),
    )
    # Align labels
    fig.align_labels()
    # Save targets
    for target in targets:
        fig.savefig(target, **SAVEFIG_PARAMS)


def soft_robot_eig(dependencies: List[pathlib.Path],
                   targets: List[pathlib.Path]) -> None:
    """Save soft robot eigenvalue plot."""
    deps = _open_hydra_pickles(dependencies)
    edmd = deps['soft_robot__polynomial3_delay1__edmd']
    srconst = deps['soft_robot__polynomial3_delay1__srconst_0999']
    hinf = deps['soft_robot__polynomial3_delay1__hinf']
    hinfw = deps['soft_robot__polynomial3_delay1__hinfw']
    # Create figure
    fig = plt.figure(constrained_layout=True, figsize=(10, 5))
    ax = fig.add_subplot(projection='polar')
    # Plot unit circle
    th = np.linspace(0, 2 * np.pi)
    ax.plot(th, np.ones(th.shape), '--', color='k', linewidth=1.5)
    # Shared style for scatter plots
    style = {
        's': 50,
        'edgecolors': 'w',
        'linewidth': 0.25,
        'zorder': 2,
    }
    # Plot eigenvalues
    ax.scatter(
        np.angle(edmd['eigenvalues']['eigv']),
        np.absolute(edmd['eigenvalues']['eigv']),
        color=C['edmd'],
        marker='o',
        label=r'Extended DMD',
        **style,
    )
    ax.scatter(
        np.angle(srconst['eigenvalues']['eigv']),
        np.absolute(srconst['eigenvalues']['eigv']),
        color=C['srconst'],
        marker='s',
        label=r'A.S. constraint',
        **style,
    )
    ax.scatter(
        np.angle(hinf['eigenvalues']['eigv']),
        np.absolute(hinf['eigenvalues']['eigv']),
        color=C['hinf'],
        marker='D',
        label=f'{HINF} regularizer',
        **style,
    )
    # Create sub-axes for zoomed plot
    axins = fig.add_axes([0.6, 0.05, 0.5, 0.5], projection='polar')
    # Plot unit circle in zoomed plot
    axins.plot(th, np.ones(th.shape), '--', color='k', linewidth=1.5)
    # Set limits for zoomed plot
    rmax = 1.05
    thmax = np.pi / 16
    axins.set_rlim(0, rmax)
    axins.set_thetalim(-thmax, thmax)
    # Plot eigenvalues in zoomed plot
    axins.scatter(
        np.angle(edmd['eigenvalues']['eigv']),
        np.absolute(edmd['eigenvalues']['eigv']),
        color=C['edmd'],
        marker='o',
        label=r'Extended DMD',
        **style,
    )
    axins.scatter(
        np.angle(srconst['eigenvalues']['eigv']),
        np.absolute(srconst['eigenvalues']['eigv']),
        color=C['srconst'],
        marker='s',
        label=r'A.S. constraint',
        **style,
    )
    axins.scatter(
        np.angle(hinf['eigenvalues']['eigv']),
        np.absolute(hinf['eigenvalues']['eigv']),
        color=C['hinf'],
        marker='D',
        label=f'{HINF} regularizer',
        **style,
    )
    # Border line width and color
    border_lw = 1
    border_color = 'k'
    # Plot border of zoomed area
    thb = np.linspace(-thmax, thmax, 1000)
    ax.plot(thb, rmax * np.ones_like(thb), border_color, linewidth=border_lw)
    rb = np.linspace(0, rmax, 1000)
    ax.plot(thmax * np.ones_like(rb), rb, border_color, linewidth=border_lw)
    ax.plot(-thmax * np.ones_like(rb), rb, border_color, linewidth=border_lw)
    # Create lines linking border to zoomed plot
    axins.annotate(
        '',
        xy=(thmax, rmax),
        xycoords=ax.transData,
        xytext=(thmax, rmax),
        textcoords=axins.transData,
        arrowprops={
            'arrowstyle': '-',
            'linewidth': border_lw,
            'color': border_color,
            'shrinkA': 0,
            'shrinkB': 0,
        },
    )
    axins.annotate(
        '',
        xy=(-thmax, 0),
        xycoords=ax.transData,
        xytext=(-thmax, 0),
        textcoords=axins.transData,
        arrowprops={
            'arrowstyle': '-',
            'linewidth': border_lw,
            'color': border_color,
            'shrinkA': 0,
            'shrinkB': 0,
        },
    )
    # Create legend
    ax.legend(loc='lower left', ncol=1)
    # Set axis limits and ticks
    ax.set_rlim(0, 2.5)
    ax.set_yticks([0, 0.5, 1, 1.5, 2, 2.5])
    # Set axis labels
    ax.set_xlabel(r'$\mathrm{Re}\{\lambda_i\}$')
    ax.set_ylabel(r'$\mathrm{Im}\{\lambda_i\}$', labelpad=25)
    # Save targets
    for target in targets:
        fig.savefig(target, **SAVEFIG_PARAMS)


def soft_robot_bode(dependencies: List[pathlib.Path],
                    targets: List[pathlib.Path]) -> None:
    """Save soft robot bode plot."""
    deps = _open_hydra_pickles(dependencies)
    edmd = deps['soft_robot__polynomial3_delay1__edmd']
    srconst = deps['soft_robot__polynomial3_delay1__srconst_0999']
    hinf = deps['soft_robot__polynomial3_delay1__hinf']
    hinfw = deps['soft_robot__polynomial3_delay1__hinfw']
    # Create figure
    fig, ax = plt.subplots(constrained_layout=True, figsize=(5, 5))
    # Plot magnitude response
    ax.semilogx(
        edmd['bode']['f_plot'],
        edmd['bode']['mag_db'],
        label='Extended DMD',
        color=C['edmd'],
    )
    ax.semilogx(
        srconst['bode']['f_plot'],
        srconst['bode']['mag_db'],
        label='A.S. constraint',
        color=C['srconst'],
    )
    ax.semilogx(
        hinf['bode']['f_plot'],
        hinf['bode']['mag_db'],
        label=f'{HINF} regularizer',
        color=C['hinf'],
    )
    # Create legend
    ax.legend(loc='upper right')
    # Set axis labels and limits
    ax.set_xlabel(r'$f$ (Hz)')
    ax.set_ylabel(r'$\bar{\sigma}\left({\bf G}(e^{j \theta})\right)$ (dB)')
    ax.set_ylim(10, 150)
    # Save targets
    for target in targets:
        fig.savefig(target, **SAVEFIG_PARAMS)


def soft_robot_svd(dependencies: List[pathlib.Path],
                   targets: List[pathlib.Path]) -> None:
    """Save soft robot SVD plot."""
    deps = _open_hydra_pickles(dependencies)
    edmd = deps['soft_robot__polynomial3_delay1__edmd']
    srconst = deps['soft_robot__polynomial3_delay1__srconst_0999']
    hinf = deps['soft_robot__polynomial3_delay1__hinf']
    hinfw = deps['soft_robot__polynomial3_delay1__hinfw']
    # Calculate singular values of ``A`` and ``B``
    sv_A_edmd, sv_B_edmd = _calc_sv(edmd['matshow']['U'])
    sv_A_srconst, sv_B_srconst = _calc_sv(srconst['matshow']['U'])
    sv_A_hinf, sv_B_hinf = _calc_sv(hinf['matshow']['U'])
    sv_A_hinfw, sv_B_hinfw = _calc_sv(hinfw['matshow']['U'])
    # Create figure
    fig, ax = plt.subplots(
        1,
        2,
        constrained_layout=True,
        sharey=True,
        figsize=(10, 5),
    )
    # Plot singular values of ``A``
    ax[0].semilogy(sv_A_edmd, marker='.', color=C['edmd'])
    ax[0].semilogy(sv_A_srconst, marker='.', color=C['srconst'])
    ax[0].semilogy(sv_A_hinf, marker='.', color=C['hinf'])
    # Plot singular values of ``B``
    ax[1].semilogy(
        sv_B_edmd,
        label='Extended DMD',
        marker='.',
        color=C['edmd'],
    )
    ax[1].semilogy(
        sv_B_srconst,
        label='A.S. constraint',
        marker='.',
        color=C['srconst'],
    )
    ax[1].semilogy(
        sv_B_hinf,
        label=f'{HINF} regularizer',
        marker='.',
        color=C['hinf'],
    )
    # Set axis limits and ticks
    ax[0].set_ylim(10**-6, 10**4)
    ax[0].set_yticks([10**n for n in range(-6, 5)])
    # Create legend
    ax[1].legend(loc='lower right')
    # Set axis labels
    ax[0].set_xlabel(r'$i$')
    ax[0].set_ylabel(r'$\sigma_i(\bf{A})$')
    ax[1].set_xlabel(r'$i$')
    ax[1].set_ylabel(r'$\sigma_i(\bf{B})$')
    # Save targets
    for target in targets:
        fig.savefig(target, **SAVEFIG_PARAMS)


def soft_robot_weights(dependencies: List[pathlib.Path],
                       targets: List[pathlib.Path]) -> None:
    """Save soft robot bode weights."""
    deps = _open_hydra_pickles(dependencies)
    hinf = deps['soft_robot__polynomial3_delay1__hinf']
    hinfw = deps['soft_robot__polynomial3_delay1__hinfw']
    # Create figure
    fig, ax1 = plt.subplots(constrained_layout=True, figsize=(5, 5))
    # Create right axis
    ax2 = ax1.twinx()
    # Plot frequency responses
    ax1.semilogx(
        hinf['bode']['f_plot'],
        hinf['bode']['mag_db'],
        label=f'{HINF} regularizer',
        color=C['hinf'],
    )
    ax1.semilogx(
        hinfw['bode']['f_plot'],
        hinfw['bode']['mag_db'],
        label=f'W. {HINF} reg.',
        color=C['hinfw'],
    )
    ax2.semilogx(
        hinfw['weights']['w_dt'] / 2 / np.pi * hinfw['bode']['f_samp'],
        hinfw['weights']['mag_dt_db'],
        '--',
        label=r'Weight',
        color=C['hinfw_weight'],
    )
    # Set axis labels
    ax1.set_xlabel('$f$ (Hz)')
    ax1.set_ylabel(r'$\bar{\sigma}\left({\bf G}(e^{j \theta})\right)$ (dB)')
    ax2.set_ylabel(r'Weight magnitude (dB)')
    # Set axis limits
    b1 = 14  # Lower limit of right axis
    b2 = -4  # Lower limit of left axis
    n = 16  # Number of dB in the axis limits
    ax1.set_ylim(b1, b1 + n)
    ax2.set_ylim(b2, b2 + n)
    # Set ticks, making sure they're the same for both axes
    loc1 = matplotlib.ticker.LinearLocator(numticks=((n // 2) + 1))
    loc2 = matplotlib.ticker.LinearLocator(numticks=((n // 2) + 1))
    ax1.yaxis.set_major_locator(loc1)
    ax2.yaxis.set_major_locator(loc2)
    # Add legends
    # https://stackoverflow.com/questions/25829736/matplotlib-how-to-adjust-zorder-of-second-legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    first_legend = plt.legend(
        handles1,
        labels1,
        loc='upper left',
        title=r'\textbf{Left axis}',
    )
    ax2.add_artist(first_legend)
    plt.legend(
        handles2,
        labels2,
        loc='upper right',
        title=r'\textbf{Right axis}',
    )
    # Save targets
    for target in targets:
        fig.savefig(target, **SAVEFIG_PARAMS)


def soft_robot_scatter_by_method(dependencies: List[pathlib.Path],
                                 targets: List[pathlib.Path]) -> None:
    """Save soft robot bar chart grouped by method."""
    deps = _open_hydra_pickles(dependencies)
    edmd = deps['soft_robot__polynomial3_delay1__edmd']
    srconst = deps['soft_robot__polynomial3_delay1__srconst_0999']
    hinf = deps['soft_robot__polynomial3_delay1__hinf']
    hinfw = deps['soft_robot__polynomial3_delay1__hinfw']
    # Construct dataframe with RMS errors
    errors = pandas.DataFrame({
        'EDMD': _calc_rmse(edmd),
        'A.S. constr.': _calc_rmse(srconst),
        f'{HINF} reg.': _calc_rmse(hinf),
        f'W. {HINF} reg.': _calc_rmse(hinfw),
    })
    means = errors.mean()
    std = errors.std()
    # Create figure
    fig, ax = plt.subplots(constrained_layout=True, figsize=(5, 5))
    # Column colors
    c = [C['edmd'], C['srconst'], C['hinf'], C['hinfw']]
    # Mean and shifted tick locations
    x = np.array([0, 1, 2, 3])
    xm = x - 0.05
    xp = x + 0.05
    # Plot error bars
    ax.errorbar(
        xp,
        means,
        std,
        fmt='.',
        linewidth=1.5,
        color='k',
        zorder=2,
        label=r'Mean \& S.D.',
    )
    # Shared scatter plot style
    style = {
        's': 50 * 1.5,
        'edgecolors': 'w',
        'linewidth': 0.25 * 1.5,
        'zorder': 2,
    }
    # Plot scatter plots
    ax.scatter(x=xm, y=errors.iloc[0, :], c=c, marker='o', **style)
    ax.scatter(x=xm, y=errors.iloc[1, :], c=c, marker='s', **style)
    ax.scatter(x=xm, y=errors.iloc[2, :], c=c, marker='D', **style)
    ax.scatter(x=xm, y=errors.iloc[3, :], c=c, marker='P', **style)
    # Plot invisible points for use in legend
    ax.scatter(x=-1, y=-1, c='k', marker='o', label=r'Valid. ep. \#1', **style)
    ax.scatter(x=-1, y=-1, c='k', marker='s', label=r'Valid. ep. \#2', **style)
    ax.scatter(x=-1, y=-1, c='k', marker='D', label=r'Valid. ep. \#3', **style)
    ax.scatter(x=-1, y=-1, c='k', marker='P', label=r'Valid. ep. \#4', **style)
    # Set labels
    ax.set_xlabel('Regression method')
    ax.set_ylabel('RMS Euclidean error (cm)')
    # Set limits and ticks
    ax.set_ylim(0, 1.6)
    ax.set_xlim(-0.5, 3.5)
    ax.set_xticks(x)
    ax.set_xticklabels([errors.columns[i] for i in range(len(x))])
    # Create legend
    ax.legend(loc='upper right')
    # Save targets
    for target in targets:
        fig.savefig(target, **SAVEFIG_PARAMS)


def soft_robot_scatter_dmdc(dependencies: List[pathlib.Path],
                            targets: List[pathlib.Path]) -> None:
    """Save soft robot bar chart grouped by method."""
    deps = _open_hydra_pickles(dependencies)
    srconst = deps['soft_robot__polynomial3_delay1__srconst_0999']
    srconst_dmdc = deps['soft_robot__polynomial3_delay1__srconst_0999_dmdc']
    hinf = deps['soft_robot__polynomial3_delay1__hinf']
    hinf_dmdc = deps['soft_robot__polynomial3_delay1__hinf_dmdc']
    # Construct dataframe with RMS errors
    errors = pandas.DataFrame({
        'EDMD,\nA.S. constr.': _calc_rmse(srconst),
        'DMDc,\nA.S. constr.': _calc_rmse(srconst_dmdc),
        f'EDMD,\n{HINF} reg.': _calc_rmse(hinf),
        f'DMDc,\n{HINF} reg.': _calc_rmse(hinf_dmdc),
    })
    means = errors.mean()
    std = errors.std()
    # Create figure
    fig, ax = plt.subplots(constrained_layout=True, figsize=(5, 5))
    # Column colors
    c = [C['srconst'], C['srconst_dmdc'], C['hinf'], C['hinf_dmdc']]
    # Mean and shifted tick locations
    x = np.array([0, 1, 2, 3])
    xm = x - 0.05
    xp = x + 0.05
    # Plot error bars
    ax.errorbar(
        xp,
        means,
        std,
        fmt='.',
        linewidth=1.5,
        color='k',
        zorder=2,
        label=r'Mean \& S.D.',
    )
    # Shared scatter plot style
    style = {
        's': 50 * 1.5,
        'edgecolors': 'w',
        'linewidth': 0.25 * 1.5,
        'zorder': 2,
    }
    # Plot scatter plots
    ax.scatter(x=xm, y=errors.iloc[0, :], c=c, marker='o', **style)
    ax.scatter(x=xm, y=errors.iloc[1, :], c=c, marker='s', **style)
    ax.scatter(x=xm, y=errors.iloc[2, :], c=c, marker='D', **style)
    ax.scatter(x=xm, y=errors.iloc[3, :], c=c, marker='P', **style)
    # Plot invisible points for use in legend
    ax.scatter(x=-1, y=-1, c='k', marker='o', label=r'Valid. ep. \#1', **style)
    ax.scatter(x=-1, y=-1, c='k', marker='s', label=r'Valid. ep. \#2', **style)
    ax.scatter(x=-1, y=-1, c='k', marker='D', label=r'Valid. ep. \#3', **style)
    ax.scatter(x=-1, y=-1, c='k', marker='P', label=r'Valid. ep. \#4', **style)
    # Set labels
    ax.set_xlabel('Regression method')
    ax.set_ylabel('RMS Euclidean error (cm)')
    # Set limits and ticks
    ax.set_ylim(0, 2.25)
    ax.set_xlim(-0.5, 3.5)
    ax.set_xticks(x)
    ax.set_xticklabels([errors.columns[i] for i in range(len(x))])
    # Create legend
    ax.legend(loc='upper right')
    # Save targets
    for target in targets:
        fig.savefig(target, **SAVEFIG_PARAMS)


def soft_robot_dmdc_svd(dependencies: List[pathlib.Path],
                        targets: List[pathlib.Path]) -> None:
    """Save soft robot DMDc SVD plot."""
    deps = _open_hydra_pickles(dependencies)
    srconst = deps['soft_robot__polynomial3_delay1__srconst_0999']
    srconst_dmdc = deps['soft_robot__polynomial3_delay1__srconst_0999_dmdc']
    hinf = deps['soft_robot__polynomial3_delay1__hinf']
    hinf_dmdc = deps['soft_robot__polynomial3_delay1__hinf_dmdc']
    # Calculate singular values of ``A`` and ``B``
    sv_A_srconst, sv_B_srconst = _calc_sv(srconst['matshow']['U'])
    sv_A_hinf, sv_B_hinf = _calc_sv(hinf['matshow']['U'])
    sv_A_hinf_dmdc, sv_B_hinf_dmdc = _calc_sv(hinf_dmdc['matshow']['U'])
    sv_A_srconst_dmdc, sv_B_srconst_dmdc = _calc_sv(
        srconst_dmdc['matshow']['U'])
    # Create figure
    fig, ax = plt.subplots(
        1,
        2,
        constrained_layout=True,
        sharey=True,
        figsize=(10, 5),
    )
    # Plot singular values of ``A``
    ax[0].semilogy(sv_A_srconst, marker='.', color=C['srconst'])
    ax[0].semilogy(sv_A_hinf, marker='.', color=C['hinf'])
    ax[0].semilogy(sv_A_srconst_dmdc, marker='.', color=C['srconst_dmdc'])
    ax[0].semilogy(sv_A_hinf_dmdc, marker='.', color=C['hinf_dmdc'])
    # Plot singular values of ``B``
    ax[1].semilogy(
        sv_B_srconst,
        label='EDMD, A.S. constr.',
        marker='.',
        color=C['srconst'],
    )
    ax[1].semilogy(
        sv_B_hinf,
        label=f'EDMD, {HINF} reg.',
        marker='.',
        color=C['hinf'],
    )
    ax[1].semilogy(
        sv_B_srconst_dmdc,
        label='DMDc, A.S. constr.',
        marker='.',
        color=C['srconst_dmdc'],
    )
    ax[1].semilogy(
        sv_B_hinf_dmdc,
        label=f'DMDc, {HINF} reg.',
        marker='.',
        color=C['hinf_dmdc'],
    )
    # Set axis limits and ticks
    ax[0].set_ylim(10**-6, 10**4)
    ax[0].set_yticks([10**n for n in range(-6, 5)])
    # Create legend
    ax[1].legend(loc='lower right')
    # Set axis labels
    ax[0].set_xlabel(r'$i$')
    ax[0].set_ylabel(r'$\sigma_i(\bf{A})$')
    ax[1].set_xlabel(r'$i$')
    ax[1].set_ylabel(r'$\sigma_i(\bf{B})$')
    # Save targets
    for target in targets:
        fig.savefig(target, **SAVEFIG_PARAMS)


def soft_robot_dmdc_bode(dependencies: List[pathlib.Path],
                         targets: List[pathlib.Path]) -> None:
    """Save soft robot DMDc bode plot."""
    deps = _open_hydra_pickles(dependencies)
    srconst = deps['soft_robot__polynomial3_delay1__srconst_0999']
    srconst_dmdc = deps['soft_robot__polynomial3_delay1__srconst_0999_dmdc']
    hinf = deps['soft_robot__polynomial3_delay1__hinf']
    hinf_dmdc = deps['soft_robot__polynomial3_delay1__hinf_dmdc']
    # Create figure
    fig, ax = plt.subplots(constrained_layout=True, figsize=(5, 5))
    ax.semilogx(
        srconst['bode']['f_plot'],
        srconst['bode']['mag_db'],
        label='EDMD, A.S. constr.',
        color=C['srconst'],
    )
    ax.semilogx(
        hinf['bode']['f_plot'],
        hinf['bode']['mag_db'],
        label=f'EDMD, {HINF} reg.',
        color=C['hinf'],
    )
    ax.semilogx(
        srconst_dmdc['bode']['f_plot'],
        srconst_dmdc['bode']['mag_db'],
        label='DMDc, A.S. constr.',
        color=C['srconst_dmdc'],
    )
    ax.semilogx(
        hinf_dmdc['bode']['f_plot'],
        hinf_dmdc['bode']['mag_db'],
        label=f'DMDc, {HINF} reg.',
        color=C['hinf_dmdc'],
    )
    # Create legend
    ax.legend(loc='upper right')
    # Set axis labels and limits
    ax.set_xlabel('$f$ (Hz)')
    ax.set_ylabel(r'$\bar{\sigma}\left({\bf G}(e^{j \theta})\right)$ (dB)')
    ax.set_ylim(10, 150)
    # Save targets
    for target in targets:
        fig.savefig(target, **SAVEFIG_PARAMS)


def soft_robot_ram(dependencies: List[pathlib.Path],
                   targets: List[pathlib.Path]) -> None:
    """Save soft robot performance plot."""
    deps = _open_dat_files(dependencies)
    srconst = deps['srconst_0999']
    srconst_dmdc = deps['srconst_0999_dmdc']
    hinf = deps['hinf']
    hinf_dmdc = deps['hinf_dmdc']
    # Create dataframe
    stats = pandas.DataFrame({
        'label': [
            'EDMD,\nA.S. constr.',
            'DMDc,\nA.S. constr.',
            f'EDMD,\n{HINF} reg.',
            f'DMDc,\n{HINF} reg.',
        ],
        'ram': [
            srconst[0],
            srconst_dmdc[0],
            hinf[0],
            hinf_dmdc[0],
        ],
    })
    # Plot dataframe
    fig, ax = plt.subplots(constrained_layout=True, figsize=(5, 5))
    stats.plot(
        x='label',
        y='ram',
        kind='bar',
        ax=ax,
        rot=0,
        color=[
            C['srconst'],
            C['srconst_dmdc'],
            C['hinf'],
            C['hinf_dmdc'],
        ],
        legend=False,
        zorder=2,
    )
    # Set grid only on ``x`` axis
    ax.grid(axis='x')
    # Set axis labels
    ax.set_xlabel('Regression method')
    ax.set_ylabel('Peak memory consumption (GiB)')
    # Save targets
    for target in targets:
        fig.savefig(target, **SAVEFIG_PARAMS)


def soft_robot_exec(dependencies: List[pathlib.Path],
                    targets: List[pathlib.Path]) -> None:
    """Save soft robot performance plot."""
    deps = _open_dat_files(dependencies)
    srconst = deps['srconst_0999']
    srconst_dmdc = deps['srconst_0999_dmdc']
    hinf = deps['hinf']
    hinf_dmdc = deps['hinf_dmdc']
    # Create dataframe
    stats = pandas.DataFrame({
        'label': [
            'EDMD,\nA.S. constr.',
            'DMDc,\nA.S. constr.',
            f'EDMD,\n{HINF} reg.',
            f'DMDc,\n{HINF} reg.',
        ],
        'time': [
            srconst[1],
            srconst_dmdc[1],
            hinf[1],
            hinf_dmdc[1],
        ],
    })
    # Plot dataframe
    fig, ax = plt.subplots(constrained_layout=True, figsize=(5, 5))
    stats.plot(
        x='label',
        y='time',
        kind='bar',
        ax=ax,
        rot=0,
        color=[
            C['srconst'],
            C['srconst_dmdc'],
            C['hinf'],
            C['hinf_dmdc'],
        ],
        legend=False,
        zorder=2,
    )
    # Set grid only on ``x`` axis
    ax.grid(axis='x')
    # Set axis labels
    ax.set_xlabel('Regression method')
    ax.set_ylabel('Execution time per iteration (min)')
    # Save targets
    for target in targets:
        fig.savefig(target, **SAVEFIG_PARAMS)


# --------------------------------------------------------------------------- #
# Tikz figures
# --------------------------------------------------------------------------- #


def faster_tikz_time_1a(dependencies: List[pathlib.Path],
                        targets: List[pathlib.Path]) -> None:
    """FASTER Tikz time plot 1a."""
    _faster_tikz_time_1(dependencies, targets, 0)


def faster_tikz_time_1b(dependencies: List[pathlib.Path],
                        targets: List[pathlib.Path]) -> None:
    """FASTER Tikz time plot 1b."""
    _faster_tikz_time_1(dependencies, targets, 1)


def faster_tikz_time_1c(dependencies: List[pathlib.Path],
                        targets: List[pathlib.Path]) -> None:
    """FASTER Tikz time plot 1c."""
    _faster_tikz_time_1(dependencies, targets, 2)


def _faster_tikz_time_1(dependencies: List[pathlib.Path],
                        targets: List[pathlib.Path], segment: int) -> None:
    """FASTER Tikz time plot helper 1."""
    deps = _open_hydra_pickles(dependencies)
    unconst = deps['faster__polynomial2__edmd']
    # Compute time array
    t_step = 1 / unconst['bode']['f_samp']
    n_t = int(10 / t_step)
    t = np.arange(n_t) * t_step
    # Create figure
    fig, ax = plt.subplots(constrained_layout=True, figsize=(3, 3))
    # Plot first state
    start = n_t * segment
    stop = n_t * (segment + 1)
    ax.plot(
        t,
        unconst['timeseries_1.0']['X_validation'][start:stop, 1],
        color=C['tikz_x1'],
        linewidth=3,
    )
    ax.grid(False)
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$x_1(t)$')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim([-6, 6])
    # Save targets
    for target in targets:
        fig.savefig(target, **SAVEFIG_TIKZ_PARAMS)


def faster_tikz_time_2a(dependencies: List[pathlib.Path],
                        targets: List[pathlib.Path]) -> None:
    """FASTER Tikz time plot 2a."""
    _faster_tikz_time_2(dependencies, targets, 0)


def faster_tikz_time_2b(dependencies: List[pathlib.Path],
                        targets: List[pathlib.Path]) -> None:
    """FASTER Tikz time plot 2b."""
    _faster_tikz_time_2(dependencies, targets, 1)


def faster_tikz_time_2c(dependencies: List[pathlib.Path],
                        targets: List[pathlib.Path]) -> None:
    """FASTER Tikz time plot 2c."""
    _faster_tikz_time_2(dependencies, targets, 2)


def _faster_tikz_time_2(dependencies: List[pathlib.Path],
                        targets: List[pathlib.Path], segment: int) -> None:
    """FASTER Tikz time plot helper 2."""
    deps = _open_hydra_pickles(dependencies)
    unconst = deps['faster__polynomial2__edmd']
    # Compute time array
    t_step = 1 / unconst['bode']['f_samp']
    n_t = int(10 / t_step)
    t = np.arange(n_t) * t_step
    # Create figure
    fig, ax = plt.subplots(constrained_layout=True, figsize=(3, 3))
    # Plot first state
    start = n_t * segment
    stop = n_t * (segment + 1)
    ax.plot(
        t,
        # Second state is approx 5x bigger than first
        5 * unconst['timeseries_1.0']['X_validation'][start:stop, 2],
        color=C['tikz_x2'],
        linewidth=3,
    )
    ax.grid(False)
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$x_2(t)$')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim([-6, 6])
    # Save targets
    for target in targets:
        fig.savefig(target, **SAVEFIG_TIKZ_PARAMS)


def faster_tikz_time_3a(dependencies: List[pathlib.Path],
                        targets: List[pathlib.Path]) -> None:
    """FASTER Tikz time plot 3a."""
    _faster_tikz_time_3(dependencies, targets, 0)


def faster_tikz_time_3b(dependencies: List[pathlib.Path],
                        targets: List[pathlib.Path]) -> None:
    """FASTER Tikz time plot 3b."""
    _faster_tikz_time_3(dependencies, targets, 1)


def faster_tikz_time_3c(dependencies: List[pathlib.Path],
                        targets: List[pathlib.Path]) -> None:
    """FASTER Tikz time plot 3c."""
    _faster_tikz_time_3(dependencies, targets, 2)


def _faster_tikz_time_3(dependencies: List[pathlib.Path],
                        targets: List[pathlib.Path], segment: int) -> None:
    """FASTER Tikz time plot helper 3."""
    deps = _open_hydra_pickles(dependencies)
    unconst = deps['faster__polynomial2__edmd']
    # Compute time array
    t_step = 1 / unconst['bode']['f_samp']
    n_t = int(10 / t_step)
    t = np.arange(n_t) * t_step
    # Create figure
    fig, ax = plt.subplots(constrained_layout=True, figsize=(3, 3))
    # Plot first state
    start = n_t * segment
    stop = n_t * (segment + 1)
    ax.plot(
        t,
        # Input is approx 3x bigger than first state
        3 * unconst['timeseries_1.0']['X_validation'][start:stop, 3],
        color=C['tikz_u'],
        linewidth=3,
    )
    ax.grid(False)
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$u(t)$')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim([-6, 6])
    # Save targets
    for target in targets:
        fig.savefig(target, **SAVEFIG_TIKZ_PARAMS)


def faster_tikz_lf_1(dependencies: List[pathlib.Path],
                     targets: List[pathlib.Path]) -> None:
    """FASTER Tikz lifting function plot."""
    # Create figure
    fig = plt.figure(constrained_layout=True, figsize=(3, 3))
    ax = fig.add_subplot(projection='3d')
    x, y = np.meshgrid(
        np.linspace(-1, 1, 20),
        np.linspace(-1, 1, 20),
    )
    z = y
    ax.plot_surface(x, y, z, cmap=cmcrameri.cm.batlow)
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_zlabel(r'$\psi_2(x_1, x_2, u)$')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_zlim([-1.1, 1.1])
    # Save targets
    for target in targets:
        fig.savefig(target, **SAVEFIG_TIKZ_PARAMS)


def faster_tikz_lf_2(dependencies: List[pathlib.Path],
                     targets: List[pathlib.Path]) -> None:
    """FASTER Tikz lifting function plot."""
    # Create figure
    fig = plt.figure(constrained_layout=True, figsize=(3, 3))
    ax = fig.add_subplot(projection='3d')
    x, y = np.meshgrid(
        np.linspace(-1, 1, 20),
        np.linspace(-1, 1, 20),
    )
    z = x**2
    ax.plot_surface(x, y, z, cmap=cmcrameri.cm.batlow)
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_zlabel(r'$\psi_3(x_1, x_2, u)$')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_zlim([-1.1, 1.1])
    # Save targets
    for target in targets:
        fig.savefig(target, **SAVEFIG_TIKZ_PARAMS)


def faster_tikz_lf_3(dependencies: List[pathlib.Path],
                     targets: List[pathlib.Path]) -> None:
    """FASTER Tikz lifting function plot."""
    # Create figure
    fig = plt.figure(constrained_layout=True, figsize=(3, 3))
    ax = fig.add_subplot(projection='3d')
    x, y = np.meshgrid(
        np.linspace(-1, 1, 20),
        np.linspace(-1, 1, 20),
    )
    z = x * y
    ax.plot_surface(x, y, z, cmap=cmcrameri.cm.batlow)
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_zlabel(r'$\psi_4(x_1, x_2, u)$')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_zlim([-1.1, 1.1])
    # Save targets
    for target in targets:
        fig.savefig(target, **SAVEFIG_TIKZ_PARAMS)


def faster_tikz_lifted_1a(dependencies: List[pathlib.Path],
                          targets: List[pathlib.Path]) -> None:
    """FASTER Tikz lifted time plot 1a."""
    _faster_tikz_lifted_1(dependencies, targets, 0)


def faster_tikz_lifted_1b(dependencies: List[pathlib.Path],
                          targets: List[pathlib.Path]) -> None:
    """FASTER Tikz lifted time plot 1b."""
    _faster_tikz_lifted_1(dependencies, targets, 1)


def faster_tikz_lifted_1c(dependencies: List[pathlib.Path],
                          targets: List[pathlib.Path]) -> None:
    """FASTER Tikz lifted time plot 1c."""
    _faster_tikz_lifted_1(dependencies, targets, 2)


def _faster_tikz_lifted_1(dependencies: List[pathlib.Path],
                          targets: List[pathlib.Path], segment: int) -> None:
    """FASTER Tikz lifted time plot helper 1."""
    deps = _open_hydra_pickles(dependencies)
    unconst = deps['faster__polynomial2__edmd']
    # Compute time array
    t_step = 1 / unconst['bode']['f_samp']
    n_t = int(10 / t_step)
    t = np.arange(n_t) * t_step
    # Create figure
    fig, ax = plt.subplots(constrained_layout=True, figsize=(2, 2))
    # Plot first state
    start = n_t * segment
    stop = n_t * (segment + 1)
    lf = 5 * unconst['timeseries_1.0']['X_validation'][start:stop, 2]
    carr = cmcrameri.cm.batlow(lf / 5 + 0.5)
    ax.scatter(t, lf, c=carr, s=3)
    ax.grid(False)
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$\psi_2(t)$')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim([-6, 6])
    ax.set_xlim([0, t_step * t.size])
    # Save targets
    for target in targets:
        fig.savefig(target, **SAVEFIG_TIKZ_PARAMS)


def faster_tikz_lifted_2a(dependencies: List[pathlib.Path],
                          targets: List[pathlib.Path]) -> None:
    """FASTER Tikz lifted time plot 2a."""
    _faster_tikz_lifted_2(dependencies, targets, 0)


def faster_tikz_lifted_2b(dependencies: List[pathlib.Path],
                          targets: List[pathlib.Path]) -> None:
    """FASTER Tikz lifted time plot 2b."""
    _faster_tikz_lifted_2(dependencies, targets, 1)


def faster_tikz_lifted_2c(dependencies: List[pathlib.Path],
                          targets: List[pathlib.Path]) -> None:
    """FASTER Tikz lifted time plot 2c."""
    _faster_tikz_lifted_2(dependencies, targets, 2)


def _faster_tikz_lifted_2(dependencies: List[pathlib.Path],
                          targets: List[pathlib.Path], segment: int) -> None:
    """FASTER Tikz lifted time plot helper 2."""
    deps = _open_hydra_pickles(dependencies)
    unconst = deps['faster__polynomial2__edmd']
    # Compute time array
    t_step = 1 / unconst['bode']['f_samp']
    n_t = int(10 / t_step)
    t = np.arange(n_t) * t_step
    # Create figure
    fig, ax = plt.subplots(constrained_layout=True, figsize=(2, 2))
    # Plot first state
    start = n_t * segment
    stop = n_t * (segment + 1)
    lf = unconst['timeseries_1.0']['X_validation'][start:stop, 1]**2
    carr = cmcrameri.cm.batlow(lf / 5)
    ax.scatter(t, lf, c=carr, s=3)
    ax.grid(False)
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$\psi_3(t)$')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim([-6, 6])
    # Save targets
    for target in targets:
        fig.savefig(target, **SAVEFIG_TIKZ_PARAMS)


def faster_tikz_lifted_3a(dependencies: List[pathlib.Path],
                          targets: List[pathlib.Path]) -> None:
    """FASTER Tikz lifted time plot 3a."""
    _faster_tikz_lifted_3(dependencies, targets, 0)


def faster_tikz_lifted_3b(dependencies: List[pathlib.Path],
                          targets: List[pathlib.Path]) -> None:
    """FASTER Tikz lifted time plot 3b."""
    _faster_tikz_lifted_3(dependencies, targets, 1)


def faster_tikz_lifted_3c(dependencies: List[pathlib.Path],
                          targets: List[pathlib.Path]) -> None:
    """FASTER Tikz lifted time plot 3c."""
    _faster_tikz_lifted_3(dependencies, targets, 2)


def _faster_tikz_lifted_3(dependencies: List[pathlib.Path],
                          targets: List[pathlib.Path], segment: int) -> None:
    """FASTER Tikz lifted time plot helper 3."""
    deps = _open_hydra_pickles(dependencies)
    unconst = deps['faster__polynomial2__edmd']
    # Compute time array
    t_step = 1 / unconst['bode']['f_samp']
    n_t = int(10 / t_step)
    t = np.arange(n_t) * t_step
    # Create figure
    fig, ax = plt.subplots(constrained_layout=True, figsize=(2, 2))
    # Plot first state
    start = n_t * segment
    stop = n_t * (segment + 1)
    lf = (5 * unconst['timeseries_1.0']['X_validation'][start:stop, 1]
            * unconst['timeseries_1.0']['X_validation'][start:stop, 2])
    carr = cmcrameri.cm.batlow(lf / 5 + 0.5)
    ax.scatter(t, lf, c=carr, s=3)
    ax.grid(False)
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$\psi_4(t)$')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim([-6, 6])
    # Save targets
    for target in targets:
        fig.savefig(target, **SAVEFIG_TIKZ_PARAMS)


def faster_tikz_eig(dependencies: List[pathlib.Path],
                    targets: List[pathlib.Path]) -> None:
    """Save FASTER Tikz eigenvalue plot."""
    deps = _open_hydra_pickles(dependencies)
    const099 = deps['faster__polynomial2__srconst_099']
    # Create figure
    fig = plt.figure(constrained_layout=True, figsize=(4, 4))
    ax = fig.add_subplot(projection='polar')
    # Set common scatter plot settings
    style = {
        's': 50 * 1.5,
        'edgecolors': 'w',
        'linewidth': 0.25 * 1.5,
    }
    # Plot eigenvalue constraints
    th = np.linspace(0, 2 * np.pi)
    ax.plot(
        th,
        0.99 * np.ones(th.shape),
        '--',
        color=C['tikz_rho'],
        linewidth=3,
        zorder=2,
    )
    # Plot eigenvalues
    ax.scatter(
        np.angle(const099['eigenvalues']['eigv']),
        np.absolute(const099['eigenvalues']['eigv']),
        color=C['tikz_eig'],
        marker='o',
        zorder=2,
        **style,
    )
    # Set axis labels
    ax.set_yticks([0, 0.33, 0.66, 1])
    ax.set_yticklabels(['', '', '', '1.0'])
    ax.set_rlim([0, 1.33])
    ax.text(
        x=-np.pi / 4 + np.pi / 16,
        y=0.75,
        s=r'$\bar{\rho}$',
        color=C['tikz_rho'],
        fontsize='x-large',
    )
    # Save targets
    for target in targets:
        fig.savefig(target, **SAVEFIG_PARAMS)


def faster_tikz_bode(dependencies: List[pathlib.Path],
                     targets: List[pathlib.Path]) -> None:
    """Save FASTER Tikz eigenvalue plot."""
    deps = _open_hydra_pickles(dependencies)
    const099 = deps['faster__polynomial2__srconst_099']
    # Create figure
    fig, ax = plt.subplots(constrained_layout=True, figsize=(4, 4))
    ax.semilogx(
        const099['bode']['f_plot'],
        const099['bode']['mag_db'],
        linewidth=3,
        color=C['tikz_bode'],
    )
    ax.grid(True, linestyle='--')
    peak = np.max(const099['bode']['mag_db'])
    ax.axhline(
        y=peak,
        ls='--',
        linewidth=3,
        color=C['tikz_hinf'],
    )
    ax.text(
        x=8,
        y=peak - 6,
        s=r'$\|\boldsymbol{\mathcal{G}}\|_\infty$',
        color=C['tikz_hinf'],
        fontsize='x-large',
    )
    # Set axis labels and limits
    ax.set_xlabel(r'$f$')
    ax.set_ylabel(r'$\bar{\sigma}\left({\bf G}(f)\right)$')
    # ax.set_ylabel(r'Gain')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # Save targets
    for target in targets:
        fig.savefig(target, **SAVEFIG_PARAMS)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _calc_sv(U: np.ndarray,
             tol: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate singular values of Koopman matrix.

    Parameters
    ----------
    U: np.ndarray
        Koopman matrix.
    tol: float
        Singular value cutoff.

    Tuple[np.ndarray, np.ndarray]
        Singular values of ``A`` and ``B``.

    """
    nx = U.shape[0]
    # Extract ``A`` and ``B``
    A = U[:, :nx]
    B = U[:, nx:]
    # Compute SVDs
    sv_A = linalg.svdvals(A)
    sv_B = linalg.svdvals(B)
    return (sv_A[sv_A > tol], sv_B[sv_B > tol])


def _calc_rmse(loaded_pickle: Dict[str, Any]) -> List[float]:
    """Calculate RMS errors from a loaded results pickle.

    Parameters
    ----------
    loaded_pickle : Dict[str, Any]
        Pickle loaded from soft robot dataset.

    Returns : List[float]
        List of RMS errors.
    """
    # Timeseries to evaluate
    datasets = [f'timeseries_{n}.0' for n in ['13', '14', '15', '16']]
    rmses = []
    for ds in datasets:
        # Extract prediction and validation data
        pred = loaded_pickle[ds]['X_prediction'][:, 1:]
        vald = loaded_pickle[ds]['X_validation'][:, 1:(pred.shape[1] + 1)]
        # Compute RMS error
        err = np.linalg.norm(vald - pred, axis=1)
        rmse = np.sqrt(np.mean(err**2))
        rmses.append(rmse)
    return rmses


def _open_dat_files(
        paths: List[pathlib.Path]) -> Dict[str, Tuple[float, float]]:
    """Read a ``dat`` file and return the max RAM and execution time.

    Parameters
    ----------
    paths : List[pathlib.Path]
        List of paths to ``dat`` files generated by Memory Profiler.

    Returns
    -------
    Dict[str, Tuple[float, float]]
        Dict of peak RAM and execution time, where the key is the loaded file
        name without its extension.
    """
    loaded_data = {}
    for path in paths:
        # Load file
        name = pathlib.Path(path).stem
        with open(path, 'r') as f:
            data = f.read()
        # Define regexes
        mem_re = re.compile('MEM (.*) .*')
        func_re = re.compile('FUNC .* .* (.*) .* (.*) .*')
        # Iterate through lines
        mems = []
        times = []
        lines = data.split('\n')
        for line in lines:
            # Match regexes
            mem_match = mem_re.findall(line)
            func_match = func_re.findall(line)
            # Extract matches
            if mem_match:
                mems.append(float(mem_match[0]))
            elif func_match:
                t2 = float(func_match[0][1])
                t1 = float(func_match[0][0])
                times.append(t2 - t1)
        # Calculate stats
        max_mem = np.max(mems) / 1024  # MiB to GiB
        if len(times) == 0:
            time = 0.0
        elif len(times) == 1:
            time = times[0] / 60  # sec to min
        else:
            raise ValueError('More than one `FUNC` in `dat` file.')
        # Get file name
        loaded_data[name] = (max_mem, time)
    return loaded_data


def _open_hydra_pickles(paths: List[pathlib.Path]) -> Dict[str, Any]:
    """Open pickles in directory of Hydra log and return dict of data.

    Parameters
    ----------
    paths : List[pathlib.Path]
        Paths to Hydra pickles to load.

    Returns
    -------
    Dict[str, Any]
        Dict of loaded data, where the key is the loaded file name without its
        extension.
    """
    loaded_data = {}
    for path in paths:
        name = pathlib.Path(path).parent.name
        with open(path, 'rb') as f:
            opened_pickle = pickle.load(f)
        loaded_data[name] = opened_pickle
    return loaded_data
