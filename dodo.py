import itertools
import pathlib
import pickle
import shutil
from typing import Any, Dict, List, Tuple

import doit
import matplotlib
import numpy as np
import pandas
from matplotlib import pyplot as plt
from scipy import io, linalg

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

# SVD cutoff
TOL = 1e-12
# H-infinity LaTeX
HINF = r'$\mathcal{H}_\infty$'
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


def task_directory() -> Dict[str, Any]:
    """Create ``build`` directory and subdirectories."""
    yield {
        'name': BUILD_DIR.stem,
        'actions': [(doit.tools.create_folder, [BUILD_DIR])],
        'targets': [BUILD_DIR],
        'clean': [(shutil.rmtree, [BUILD_DIR, True])],
        'uptodate': [True],
    }
    for subdir in BUILD_DIRS.values():
        yield {
            'name': BUILD_DIR.stem + '/' + subdir.stem,
            'actions': [(doit.tools.create_folder, [subdir])],
            'task_dep': [f'directory:{BUILD_DIR.stem}'],
            'targets': [subdir],
            'clean': [(shutil.rmtree, [subdir, True])],
            'uptodate': [True],
        }


def task_pickle() -> Dict[str, Any]:
    """Pickle a dataset."""
    yield {
        'name': 'faster',
        'actions': [pickle_faster_dataset],
        'file_dep': [DATASETS_DIR.joinpath('faster/faster.csv')],
        'task_dep': ['directory:build/datasets'],
        'targets': [BUILD_DIRS['datasets'].joinpath('faster.pickle')],
        'clean': True,
    }
    yield {
        'name':
        'soft_robot',
        'actions': [pickle_soft_robot_dataset],
        'file_dep': [
            DATASETS_DIR.joinpath(
                'soft_robot/soft-robot-koopman/datafiles/softrobot_train-13_val-4.mat'
            )
        ],
        'task_dep': ['directory:build/datasets'],
        'targets': [BUILD_DIRS['datasets'].joinpath('soft_robot.pickle')],
        'clean':
        True,
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
            'task_dep': ['directory:build/hydra_outputs'],
            'targets': [exp_dir.joinpath(HYDRA_PICKLE)],
            'clean': [(shutil.rmtree, [exp_dir, True])],
        }


def task_plot() -> Dict[str, Any]:
    """Plot a figure."""
    for action in [faster_eig, faster_error]:
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
            'task_dep': [
                'directory:build/figures',
                'directory:build/cvd_figures',
            ],
            'targets': [
                BUILD_DIRS['figures'].joinpath(f'{action.__name__}.pdf'),
                BUILD_DIRS['cvd_figures'].joinpath(f'{action.__name__}.png'),
            ],
            'clean':
            True,
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
            'task_dep': [
                'directory:build/figures',
                'directory:build/cvd_figures',
            ],
            'targets': [
                BUILD_DIRS['figures'].joinpath(f'{action.__name__}.pdf'),
                BUILD_DIRS['cvd_figures'].joinpath(f'{action.__name__}.png'),
            ],
            'clean':
            True,
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
            'task_dep': [
                'directory:build/figures',
                'directory:build/cvd_figures',
            ],
            'targets': [
                BUILD_DIRS['figures'].joinpath(f'{action.__name__}.pdf'),
                BUILD_DIRS['cvd_figures'].joinpath(f'{action.__name__}.png'),
            ],
            'clean':
            True,
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
    deps = _open_hydra_pickles(dependencies)
    unconst = deps['faster__polynomial2__edmd']
    const1 = deps['faster__polynomial2__srconst_1']
    const099 = deps['faster__polynomial2__srconst_099']

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


def faster_eig(dependencies: List[pathlib.Path],
               targets: List[pathlib.Path]) -> None:
    """Save FASTER eigenvalue plot."""
    deps = _open_hydra_pickles(dependencies)
    unconst = deps['faster__polynomial2__edmd']
    const1 = deps['faster__polynomial2__srconst_1']
    const099 = deps['faster__polynomial2__srconst_099']

    fig = plt.figure(constrained_layout=True)
    ax = fig.add_subplot(projection='polar')

    scatter_style = {
        's': 50 * 1.5,
        'edgecolors': 'w',
        'linewidth': 0.25 * 1.5,
    }

    th = np.linspace(0, 2 * np.pi)
    ax.plot(
        th,
        np.ones(th.shape),
        '--',
        color=C['1.00'],
        linewidth=1.5,
    )
    ax.plot(
        th,
        0.99 * np.ones(th.shape),
        '--',
        color=C['0.99'],
        linewidth=1.5,
    )

    ax.scatter(
        np.angle(const1['eigenvalues']['eigv']),
        np.absolute(const1['eigenvalues']['eigv']),
        color=C['1.00'],
        marker='o',
        label=r'A.S. constr., $\bar{\rho} = 1.00$',
        **scatter_style,
    )
    ax.scatter(
        np.angle(const099['eigenvalues']['eigv']),
        np.absolute(const099['eigenvalues']['eigv']),
        color=C['0.99'],
        marker='s',
        label=r'A.S. constr., $\bar{\rho} = 0.99$',
        **scatter_style,
    )

    ax.text(0, 1.125, r'$\angle \lambda_i$')
    ax.text(-np.pi / 8 - np.pi / 16, 0.5, r'$|\lambda_i|$')
    ax.set_axisbelow(True)

    ax.legend(loc='lower left', ncol=1)

    ax.set_xticks([d * np.pi / 180 for d in [-20, -10, 0, 10, 20]])
    ax.set_thetalim(-np.pi / 8, np.pi / 8)

    for target in targets:
        fig.savefig(target, bbox_inches='tight', pad_inches=0.1)


def soft_robot_error(dependencies: List[pathlib.Path],
                     targets: List[pathlib.Path]) -> None:
    """Save soft robot timeseries plot."""
    deps = _open_hydra_pickles(dependencies)
    edmd = deps['soft_robot__polynomial3_delay1__edmd']
    srconst = deps['soft_robot__polynomial3_delay1__srconst_0999']
    hinf = deps['soft_robot__polynomial3_delay1__hinf']
    hinfw = deps['soft_robot__polynomial3_delay1__hinfw']

    series = 'timeseries_15.0'

    t_step = 1 / edmd['bode']['f_samp']
    n_t = edmd[series]['X_validation'].shape[0]
    t = np.arange(n_t) * t_step

    n_x = edmd[series]['X_prediction'].shape[1] - 1

    fig, ax = plt.subplots(3, 1, constrained_layout=True, sharex=True)

    for i in range(2):
        ax[i].plot(
            t,
            (edmd[series]['X_validation'][:n_t, i + 1] -
             edmd[series]['X_prediction'][:n_t, i + 1]),
            label='Extended DMD',
            color=C['edmd'],
        )
        ax[i].plot(
            t,
            (edmd[series]['X_validation'][:n_t, i + 1] -
             srconst[series]['X_prediction'][:n_t, i + 1]),
            label='A.S. constraint',
            color=C['srconst'],
        )
        ax[i].plot(
            t,
            (edmd[series]['X_validation'][:n_t, i + 1] -
             hinf[series]['X_prediction'][:n_t, i + 1]),
            label=f'{HINF} regularizer',
            color=C['hinf'],
        )

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

    ax[-1].set_xlabel(r'$t$ (s)')

    ax[0].set_ylabel(r'$\Delta x_1(t)$ (cm)')
    ax[1].set_ylabel(r'$\Delta x_2(t)$ (cm)')
    ax[2].set_ylabel(r'${\bf u}(t)$ (V)')

    ax[0].set_ylim(-5, 5)
    ax[1].set_ylim(-5, 5)
    ax[2].set_ylim(-1, 9)

    ax[0].set_yticks([-4, -2, 0, 2, 4])
    ax[1].set_yticks([-4, -2, 0, 2, 4])
    ax[2].set_yticks([0, 2, 4, 6, 8])

    ax[2].legend(
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
        bbox_to_anchor=(0.5, -0.5),
    )

    fig.align_labels()

    for target in targets:
        fig.savefig(target, bbox_inches='tight', pad_inches=0.1)


def soft_robot_eig(dependencies: List[pathlib.Path],
                   targets: List[pathlib.Path]) -> None:
    """Save soft robot eigenvalue plot."""
    deps = _open_hydra_pickles(dependencies)
    edmd = deps['soft_robot__polynomial3_delay1__edmd']
    srconst = deps['soft_robot__polynomial3_delay1__srconst_0999']
    hinf = deps['soft_robot__polynomial3_delay1__hinf']
    hinfw = deps['soft_robot__polynomial3_delay1__hinfw']

    fig = plt.figure(constrained_layout=True)
    ax = fig.add_subplot(projection='polar')

    th = np.linspace(0, 2 * np.pi)
    ax.plot(
        th,
        np.ones(th.shape),
        '--',
        color='k',
        linewidth=1.5,
    )

    scatter_style = {
        's': 50,
        'edgecolors': 'w',
        'linewidth': 0.25,
        'zorder': 2,
    }

    ax.scatter(
        np.angle(edmd['eigenvalues']['eigv']),
        np.absolute(edmd['eigenvalues']['eigv']),
        color=C['edmd'],
        marker='o',
        label=r'Extended DMD',
        **scatter_style,
    )
    ax.scatter(
        np.angle(srconst['eigenvalues']['eigv']),
        np.absolute(srconst['eigenvalues']['eigv']),
        color=C['srconst'],
        marker='s',
        label=r'A.S. constraint',
        **scatter_style,
    )
    ax.scatter(
        np.angle(hinf['eigenvalues']['eigv']),
        np.absolute(hinf['eigenvalues']['eigv']),
        color=C['hinf'],
        marker='D',
        label=f'{HINF} regularizer',
        **scatter_style,
    )

    axins = fig.add_axes(
        [0.6, 0.05, 0.5, 0.5],
        projection='polar',
    )

    axins.plot(
        th,
        np.ones(th.shape),
        '--',
        color='k',
        linewidth=1.5,
    )

    rmax = 1.05
    thmax = np.pi / 16

    axins.set_rlim(0, rmax)
    axins.set_thetalim(-thmax, thmax)

    axins.scatter(
        np.angle(edmd['eigenvalues']['eigv']),
        np.absolute(edmd['eigenvalues']['eigv']),
        color=C['edmd'],
        marker='o',
        label=r'Extended DMD',
        **scatter_style,
    )
    axins.scatter(
        np.angle(srconst['eigenvalues']['eigv']),
        np.absolute(srconst['eigenvalues']['eigv']),
        color=C['srconst'],
        marker='s',
        label=r'A.S. constraint',
        **scatter_style,
    )
    axins.scatter(
        np.angle(hinf['eigenvalues']['eigv']),
        np.absolute(hinf['eigenvalues']['eigv']),
        color=C['hinf'],
        marker='D',
        label=f'{HINF} regularizer',
        **scatter_style,
    )

    w = 1
    c = 'k'

    thb = np.linspace(-thmax, thmax, 1000)
    ax.plot(thb, rmax * np.ones_like(thb), c, linewidth=w)
    rb = np.linspace(0, rmax, 1000)
    ax.plot(thmax * np.ones_like(rb), rb, c, linewidth=w)
    ax.plot(-thmax * np.ones_like(rb), rb, c, linewidth=w)

    axins.annotate(
        '',
        xy=(thmax, rmax),
        xycoords=ax.transData,
        xytext=(thmax, rmax),
        textcoords=axins.transData,
        arrowprops={
            'arrowstyle': '-',
            'linewidth': w,
            'color': c,
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
            'linewidth': w,
            'color': c,
            'shrinkA': 0,
            'shrinkB': 0,
        },
    )

    ax.legend(loc='lower left', ncol=1)

    ax.set_rlim(0, 2.5)
    ax.set_yticks([0, 0.5, 1, 1.5, 2, 2.5])

    ax.set_xlabel(r'$\mathrm{Re}\{\lambda_i\}$')
    ax.set_ylabel(r'$\mathrm{Im}\{\lambda_i\}$', labelpad=25)

    for target in targets:
        fig.savefig(target, bbox_inches='tight', pad_inches=0.1)


def soft_robot_bode(dependencies: List[pathlib.Path],
                    targets: List[pathlib.Path]) -> None:
    """Save soft robot bode plot."""
    deps = _open_hydra_pickles(dependencies)
    edmd = deps['soft_robot__polynomial3_delay1__edmd']
    srconst = deps['soft_robot__polynomial3_delay1__srconst_0999']
    hinf = deps['soft_robot__polynomial3_delay1__hinf']
    hinfw = deps['soft_robot__polynomial3_delay1__hinfw']

    fig, ax = plt.subplots(constrained_layout=True)
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

    ax.legend(
        # handlelength=1,
        loc='upper right', )
    ax.set_xlabel(r'$f$ (Hz)')
    ax.set_ylabel(r'$\bar{\sigma}\left({\bf G}(e^{j \theta})\right)$ (dB)')
    ax.set_ylim(10, 150)

    for target in targets:
        fig.savefig(target, bbox_inches='tight', pad_inches=0.1)


def soft_robot_svd(dependencies: List[pathlib.Path],
                   targets: List[pathlib.Path]) -> None:
    """Save soft robot SVD plot."""
    deps = _open_hydra_pickles(dependencies)
    edmd = deps['soft_robot__polynomial3_delay1__edmd']
    srconst = deps['soft_robot__polynomial3_delay1__srconst_0999']
    hinf = deps['soft_robot__polynomial3_delay1__hinf']
    hinfw = deps['soft_robot__polynomial3_delay1__hinfw']

    def calc_sv(method: Dict) -> Tuple[np.ndarray, np.ndarray]:
        U = method['matshow']['U']
        nx = U.shape[0]
        A = U[:, :nx]
        B = U[:, nx:]
        sv_A = linalg.svdvals(A)
        sv_B = linalg.svdvals(B)
        return (sv_A[sv_A > TOL], sv_B[sv_B > TOL])

    sv_A_edmd, sv_B_edmd = calc_sv(edmd)
    sv_A_srconst, sv_B_srconst = calc_sv(srconst)
    sv_A_hinf, sv_B_hinf = calc_sv(hinf)
    sv_A_hinfw, sv_B_hinfw = calc_sv(hinfw)

    fig, ax = plt.subplots(1, 2, constrained_layout=True, sharey=True)

    ax[0].semilogy(
        sv_A_edmd,
        marker='.',
        color=C['edmd'],
    )
    ax[0].semilogy(
        sv_A_srconst,
        marker='.',
        color=C['srconst'],
    )
    ax[0].semilogy(
        sv_A_hinf,
        marker='.',
        color=C['hinf'],
    )
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

    ax[0].set_ylim(10**-6, 10**4)
    ax[0].set_yticks([10**n for n in range(-6, 5)])
    ax[1].legend(loc='lower right')

    ax[0].set_xlabel(r'$i$')
    ax[0].set_ylabel(r'$\sigma_i(\bf{A})$')

    ax[1].set_xlabel(r'$i$')
    ax[1].set_ylabel(r'$\sigma_i(\bf{B})$')

    for target in targets:
        fig.savefig(target, bbox_inches='tight', pad_inches=0.1)


def soft_robot_weights(dependencies: List[pathlib.Path],
                       targets: List[pathlib.Path]) -> None:
    """Save soft robot bode weights."""
    deps = _open_hydra_pickles(dependencies)
    hinf = deps['soft_robot__polynomial3_delay1__hinf']
    hinfw = deps['soft_robot__polynomial3_delay1__hinfw']

    fig, ax1 = plt.subplots(constrained_layout=True)
    ax2 = ax1.twinx()

    ax1.semilogx(
        hinf['bode']['f_plot'],
        hinf['bode']['mag_db'],
        label=f'{HINF} regularizer',
        color=C['hinf'],
    )
    ax1.semilogx(
        hinfw['bode']['f_plot'],
        hinfw['bode']['mag_db'],
        label=f'Weighted {HINF} reg.',
        color=C['hinfw'],
    )
    ax2.semilogx(
        hinfw['weights']['w_dt'] / 2 / np.pi * hinfw['bode']['f_samp'],
        hinfw['weights']['mag_dt_db'],
        '--',
        label=r'Weight',
        color=C['hinfw_weight'],
    )
    ax1.set_xlabel('$f$ (Hz)')
    ax1.set_ylabel(r'$\bar{\sigma}\left({\bf G}(e^{j \theta})\right)$ (dB)')
    ax2.set_ylabel(r'Weight magnitude (dB)')
    # ax2.spines['right'].set_color(C['hinfw_weight'])
    ax1.legend(loc='upper left', title=r'\textbf{Left axis}')
    ax2.legend(loc='upper right', title=r'\textbf{Right axis}')

    b1 = 14
    b2 = -4
    n = 16
    ax1.set_ylim(b1, b1 + n)
    ax2.set_ylim(b2, b2 + n)
    loc1 = matplotlib.ticker.LinearLocator(numticks=((n // 2) + 1))
    loc2 = matplotlib.ticker.LinearLocator(numticks=((n // 2) + 1))
    ax1.yaxis.set_major_locator(loc1)
    ax2.yaxis.set_major_locator(loc2)

    for target in targets:
        fig.savefig(target, bbox_inches='tight', pad_inches=0.1)


def soft_robot_scatter_by_method(dependencies: List[pathlib.Path],
                                 targets: List[pathlib.Path]) -> None:
    """Save soft robot bar chart grouped by method."""
    deps = _open_hydra_pickles(dependencies)
    edmd = deps['soft_robot__polynomial3_delay1__edmd']
    srconst = deps['soft_robot__polynomial3_delay1__srconst_0999']
    hinf = deps['soft_robot__polynomial3_delay1__hinf']
    hinfw = deps['soft_robot__polynomial3_delay1__hinfw']

    errors = pandas.DataFrame({
        'Extended DMD': _calc_rmse(edmd),
        'A.S. constraint': _calc_rmse(srconst),
        f'{HINF} regularizer': _calc_rmse(hinf),
        f'Weighted {HINF} reg.': _calc_rmse(hinfw),
    })
    means = errors.mean()
    std = errors.std()

    fig, ax = plt.subplots(constrained_layout=True)

    c = [C['edmd'], C['srconst'], C['hinf'], C['hinfw']]
    x = np.array([0, 1, 2, 3])
    xm = x - 0.05
    xp = x + 0.05

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

    style = {
        's': 50 * 1.5,
        'edgecolors': 'w',
        'linewidth': 0.25 * 1.5,
        'zorder': 2,
    }
    ax.scatter(x=xm, y=errors.iloc[0, :], c=c, marker='o', **style)
    ax.scatter(x=xm, y=errors.iloc[1, :], c=c, marker='s', **style)
    ax.scatter(x=xm, y=errors.iloc[2, :], c=c, marker='D', **style)
    ax.scatter(x=xm, y=errors.iloc[3, :], c=c, marker='P', **style)

    ax.scatter(x=-1, y=-1, c='k', marker='o', label=r'Valid. ep. \#1', **style)
    ax.scatter(x=-1, y=-1, c='k', marker='s', label=r'Valid. ep. \#2', **style)
    ax.scatter(x=-1, y=-1, c='k', marker='D', label=r'Valid. ep. \#3', **style)
    ax.scatter(x=-1, y=-1, c='k', marker='P', label=r'Valid. ep. \#4', **style)

    ax.set_xlabel('Regression method')
    ax.set_ylabel('RMS Euclidean error (cm)')
    ax.set_ylim(0, 1.4)
    ax.set_xticks(x)
    ax.set_xticklabels([errors.columns[i] for i in range(len(x))])

    ax.legend(loc='upper right')

    ax.set_xlim(-0.5, 3.5)

    for target in targets:
        fig.savefig(target, bbox_inches='tight', pad_inches=0.1)


def soft_robot_scatter_dmdc(dependencies: List[pathlib.Path],
                            targets: List[pathlib.Path]) -> None:
    """Save soft robot bar chart grouped by method."""
    deps = _open_hydra_pickles(dependencies)
    srconst = deps['soft_robot__polynomial3_delay1__srconst_0999']
    srconst_dmdc = deps['soft_robot__polynomial3_delay1__srconst_0999_dmdc']
    hinf = deps['soft_robot__polynomial3_delay1__hinf']
    hinf_dmdc = deps['soft_robot__polynomial3_delay1__hinf_dmdc']


    errors = pandas.DataFrame({
        'EDMD,\nA.S. constr.': _calc_rmse(srconst),
        'DMDc,\nA.S. constr.': _calc_rmse(srconst_dmdc),
        f'EDMD,\n{HINF} reg.': _calc_rmse(hinf),
        f'DMDc,\n{HINF} reg.': _calc_rmse(hinf_dmdc),
    })
    means = errors.mean()
    std = errors.std()

    fig, ax = plt.subplots(constrained_layout=True)

    c = [C['srconst'], C['srconst_dmdc'], C['hinf'], C['hinf_dmdc']]
    x = np.array([0, 1, 2, 3])
    xm = x - 0.05
    xp = x + 0.05

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

    style = {
        's': 50 * 1.5,
        'edgecolors': 'w',
        'linewidth': 0.25 * 1.5,
        'zorder': 2,
    }
    ax.scatter(x=xm, y=errors.iloc[0, :], c=c, marker='o', **style)
    ax.scatter(x=xm, y=errors.iloc[1, :], c=c, marker='s', **style)
    ax.scatter(x=xm, y=errors.iloc[2, :], c=c, marker='D', **style)
    ax.scatter(x=xm, y=errors.iloc[3, :], c=c, marker='P', **style)

    ax.scatter(x=-1, y=-1, c='k', marker='o', label=r'Valid. ep. \#1', **style)
    ax.scatter(x=-1, y=-1, c='k', marker='s', label=r'Valid. ep. \#2', **style)
    ax.scatter(x=-1, y=-1, c='k', marker='D', label=r'Valid. ep. \#3', **style)
    ax.scatter(x=-1, y=-1, c='k', marker='P', label=r'Valid. ep. \#4', **style)

    ax.set_xlabel('Regression method')
    ax.set_ylabel('RMS Euclidean error (cm)')
    ax.set_ylim(0, 2.25)
    ax.set_xticks(x)
    ax.set_xticklabels([errors.columns[i] for i in range(len(x))])

    ax.legend(loc='upper right')

    ax.set_xlim(-0.5, 3.5)

    for target in targets:
        fig.savefig(target, bbox_inches='tight', pad_inches=0.1)


def soft_robot_dmdc_svd(dependencies: List[pathlib.Path],
                        targets: List[pathlib.Path]) -> None:
    """Save soft robot DMDc SVD plot."""
    deps = _open_hydra_pickles(dependencies)
    srconst = deps['soft_robot__polynomial3_delay1__srconst_0999']
    srconst_dmdc = deps['soft_robot__polynomial3_delay1__srconst_0999_dmdc']
    hinf = deps['soft_robot__polynomial3_delay1__hinf']
    hinf_dmdc = deps['soft_robot__polynomial3_delay1__hinf_dmdc']

    def calc_sv(method: Dict) -> Tuple[np.ndarray, np.ndarray]:
        U = method['matshow']['U']
        nx = U.shape[0]
        A = U[:, :nx]
        B = U[:, nx:]
        sv_A = linalg.svdvals(A)
        sv_B = linalg.svdvals(B)
        return (sv_A[sv_A > TOL], sv_B[sv_B > TOL])

    sv_A_srconst, sv_B_srconst = calc_sv(srconst)
    sv_A_hinf, sv_B_hinf = calc_sv(hinf)
    sv_A_hinf_dmdc, sv_B_hinf_dmdc = calc_sv(hinf_dmdc)
    sv_A_srconst_dmdc, sv_B_srconst_dmdc = calc_sv(srconst_dmdc)

    fig, ax = plt.subplots(1, 2, constrained_layout=True, sharey=True)

    ax[0].semilogy(
        sv_A_srconst,
        marker='.',
        color=C['srconst'],
    )
    ax[0].semilogy(
        sv_A_hinf,
        marker='.',
        color=C['hinf'],
    )
    ax[0].semilogy(
        sv_A_srconst_dmdc,
        marker='.',
        color=C['srconst_dmdc'],
    )
    ax[0].semilogy(
        sv_A_hinf_dmdc,
        marker='.',
        color=C['hinf_dmdc'],
    )

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

    ax[0].set_ylim(10**-6, 10**4)
    ax[0].set_yticks([10**n for n in range(-6, 5)])
    ax[1].legend(loc='lower right')

    ax[0].set_xlabel(r'$i$')
    ax[0].set_ylabel(r'$\sigma_i(\bf{A})$')

    ax[1].set_xlabel(r'$i$')
    ax[1].set_ylabel(r'$\sigma_i(\bf{B})$')

    for target in targets:
        fig.savefig(target, bbox_inches='tight', pad_inches=0.1)


def soft_robot_dmdc_bode(dependencies: List[pathlib.Path],
                         targets: List[pathlib.Path]) -> None:
    """Save soft robot DMDc bode plot."""
    deps = _open_hydra_pickles(dependencies)
    srconst = deps['soft_robot__polynomial3_delay1__srconst_0999']
    srconst_dmdc = deps['soft_robot__polynomial3_delay1__srconst_0999_dmdc']
    hinf = deps['soft_robot__polynomial3_delay1__hinf']
    hinf_dmdc = deps['soft_robot__polynomial3_delay1__hinf_dmdc']

    fig, ax = plt.subplots(constrained_layout=True)
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

    ax.legend(loc='upper right')
    ax.set_xlabel('$f$ (Hz)')
    ax.set_ylabel(r'$\bar{\sigma}\left({\bf G}(e^{j \theta})\right)$ (dB)')
    ax.set_ylim(10, 150)

    for target in targets:
        fig.savefig(target, bbox_inches='tight', pad_inches=0.1)


def soft_robot_ram(dependencies: List[pathlib.Path],
                   targets: List[pathlib.Path]) -> None:
    """Save soft robot performance plot."""
    srconst = _load_dat(dependencies[0])
    srconst_dmdc = _load_dat(dependencies[1])
    hinf = _load_dat(dependencies[2])
    hinf_dmdc = _load_dat(dependencies[3])

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

    fig, ax = plt.subplots(constrained_layout=True)
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

    ax.grid(axis='x')
    ax.set_xlabel('Regression method')
    ax.set_ylabel('Peak memory consumption (GiB)')

    for target in targets:
        fig.savefig(target, bbox_inches='tight', pad_inches=0.1)


def soft_robot_exec(dependencies: List[pathlib.Path],
                    targets: List[pathlib.Path]) -> None:
    """Save soft robot performance plot."""
    srconst = _load_dat(dependencies[0])
    srconst_dmdc = _load_dat(dependencies[1])
    hinf = _load_dat(dependencies[2])
    hinf_dmdc = _load_dat(dependencies[3])

    stats = pandas.DataFrame({
        'label': [
            'EDMD,\nA.S. constr.',
            'DMDc,\nA.S. constr.',
            f'EDMD,\n{HINF} reg.',
            f'DMDc,\n{HINF} reg.',
        ],
        'time': [
            srconst[1] / 60,
            srconst_dmdc[1] / 60,
            hinf[1] / 60,
            hinf_dmdc[1] / 60,
        ],
    })

    fig, ax = plt.subplots(constrained_layout=True)
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

    ax.grid(axis='x')
    ax.set_xlabel('Regression method')
    ax.set_ylabel('Execution time per iteration (min)')

    for target in targets:
        fig.savefig(target, bbox_inches='tight', pad_inches=0.1)


def _calc_rmse(loaded_pickle):
    """Calculate RMSEs from a loaded results pickle."""
    datasets = [f'timeseries_{n}.0' for n in ['13', '14', '15', '16']]
    rmses = []
    for ds in datasets:
        pred = loaded_pickle[ds]['X_prediction'][:, 1:]
        vald = loaded_pickle[ds]['X_validation'][:, 1:(pred.shape[1] + 1)]
        err = np.linalg.norm(vald - pred, axis=1)
        rmse = np.sqrt(np.mean(err**2))
        rmses.append(rmse)
    return rmses


def _open_dat_file(path: pathlib.Path) -> Tuple[float, float]:
    """Read a ``dat`` file and return the max RAM and execution time."""
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
        time = times[0]
    else:
        raise ValueError('More than one `FUNC` in `dat` file.')

    return (max_mem, time)


def _open_hydra_pickles(paths: List[str]) -> Tuple[str, Dict[str, Any]]:
    """Open pickles in directory of Hydra log and return dict of data."""
    loaded_data = {}
    for path in paths:
        name = pathlib.Path(path).parent.name
        with open(path, 'rb') as f:
            opened_pickle = pickle.load(f)
        loaded_data[name] = opened_pickle
    return loaded_data
