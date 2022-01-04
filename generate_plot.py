"""Generate plots for paper."""

import argparse
import pathlib
import pickle
import re
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import numpy as np
import pandas
from matplotlib import pyplot as plt
from scipy import linalg

TOL = 1e-12
HINF = r'$\mathcal{H}_\infty$'
OKABE_ITO = [
    (0.00, 0.00, 0.00),  # Black
    (0.90, 0.60, 0.00),  # Orange
    (0.35, 0.70, 0.90),  # Sky Blue
    (0.00, 0.60, 0.50),  # Bluish Green
    (0.95, 0.90, 0.25),  # Yellow
    (0.00, 0.45, 0.70),  # Blue
    (0.80, 0.40, 0.00),  # Vermillion
    (0.80, 0.60, 0.70),  # Reddish Purple
]
C = {
    'edmd': OKABE_ITO[1],  # Orange
    'srconst': OKABE_ITO[2],  # Sky Blue
    'hinf': OKABE_ITO[3],  # Blusih Green
    'hinfw': OKABE_ITO[7],  # Reddish Purple
    'hinfw_weight': OKABE_ITO[5],
    #
    'srconst_dmdc': OKABE_ITO[6],  # Vermillion
    'hinf_dmdc': OKABE_ITO[4],  # Yellow
    #
    'u1': OKABE_ITO[6],
    'u2': OKABE_ITO[4],
    'u3': OKABE_ITO[5],
    #
    '1.00': OKABE_ITO[1],
    '0.99': OKABE_ITO[2],
    'u': OKABE_ITO[3],
}


def main():
    """Generate plots for paper."""
    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('figure', nargs=1)
    parser.add_argument('in_paths', nargs='+')
    parser.add_argument(
        '-o',
        '--output',
        dest='out_path',
        nargs=1,
        default=None,
    )
    parser.add_argument(
        '-n',
        '--no-latex',
        dest='no_latex',
        action='store_true',
    )
    args = parser.parse_args()
    # Get plotting function from first argument
    figure = globals()[args.figure[0]]
    # Convert paths to ``pathlib`` paths
    in_paths = [pathlib.Path(p) for p in args.in_paths]
    if args.out_path is not None:
        out_path = pathlib.Path(args.out_path[0])
    else:
        out_path = None
    # Set ``matplotlib`` settings
    plt.rc('figure', dpi=100)
    if not args.no_latex:
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=12)
    plt.rc('lines', linewidth=2)
    plt.rc('axes', grid=True)
    plt.rc('grid', linestyle='--')
    # Call figure-generating function with input and output files
    fig = figure(in_paths)
    # Save figures if plot is specified
    if out_path is not None:
        out_path.parent.mkdir(exist_ok=True)
        fig.savefig(out_path, bbox_inches='tight', pad_inches=0.1)
    else:
        plt.show()


def faster_error(in_paths: List[pathlib.Path]) -> plt.Figure:
    """Save faster timeseries plot."""
    unconst = _open_all_pickles(in_paths[0])
    const1 = _open_all_pickles(in_paths[1])
    const099 = _open_all_pickles(in_paths[2])

    t_step = 1 / unconst['bode']['f_samp']
    n_t = int(10 / t_step)
    t = np.arange(n_t) * t_step

    fig, ax = plt.subplots(3, 1, constrained_layout=True, sharex=True)
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

    # fig.set_constrained_layout_pads(
    #     h_pad=0.1,
    #     # w_pad=4 / 72,
    #     # wspace=0,
    #     # hspace=0,
    # )

    return fig


def faster_time(in_paths: List[pathlib.Path]) -> plt.Figure:
    """Save faster timeseries plot."""
    unconst = _open_all_pickles(in_paths[0])
    const1 = _open_all_pickles(in_paths[1])
    const099 = _open_all_pickles(in_paths[2])

    t_step = 1 / unconst['bode']['f_samp']
    n_t = int(10 / t_step)
    t = np.arange(n_t) * t_step

    fig, ax = plt.subplots(3, 1, constrained_layout=True, sharex=True)
    ax[0].plot(
        t,
        unconst['timeseries_1.0']['X_validation'][:n_t, 1],
        '--',
        color=C['u'],
        label='Ground truth',
    )
    ax[0].plot(
        t,
        const1['timeseries_1.0']['X_prediction'][:n_t, 1],
        label=r'A.S. constr., $\bar{\rho} = 1.00$',
        color=C['1.00'],
    )
    ax[0].plot(
        t,
        const099['timeseries_1.0']['X_prediction'][:n_t, 1],
        label=r'A.S. constr., $\bar{\rho} = 0.99$',
        color=C['0.99'],
    )

    ax[1].plot(
        t,
        unconst['timeseries_1.0']['X_validation'][:n_t, 2],
        '--',
        color=C['u'],
        label='Ground truth',
    )
    ax[1].plot(
        t,
        const1['timeseries_1.0']['X_prediction'][:n_t, 2],
        label=r'A.S. constr., $\bar{\rho} = 1.00$',
        color=C['1.00'],
    )
    ax[1].plot(
        t,
        const099['timeseries_1.0']['X_prediction'][:n_t, 2],
        label=r'A.S. constr., $\bar{\rho} = 0.99$',
        color=C['0.99'],
    )

    ax[2].plot(
        t,
        unconst['timeseries_1.0']['X_validation'][:n_t, 3],
        '--',
        color=C['u'],
        label='Ground truth',
    )

    ax[0].set_ylabel(r'$x_1(t)$ (N, norm.)')
    ax[1].set_ylabel(r'$x_2(t)$ (m, norm.)')
    ax[2].set_ylabel(r'$u_1(t)$ (V, norm.)')

    ax[2].set_xlabel(r'$t$ (s)')

    ax[2].legend(
        ax[0].get_lines(),
        [
            'Ground truth',
            r'A.S. constr., $\bar{\rho} = 1.00$',
            r'A.S. constr., $\bar{\rho} = 0.99$',
        ],
        loc='lower right',
        # handlelength=1,
    )

    ax[0].set_ylim(-2, 2)
    ax[1].set_ylim(-2, 2)
    ax[2].set_ylim(-1, 1)

    return fig


def faster_eig(in_paths: List[pathlib.Path]) -> plt.Figure:
    """Save faster eigenvalue plot."""
    unconst = _open_all_pickles(in_paths[0])
    const1 = _open_all_pickles(in_paths[1])
    const099 = _open_all_pickles(in_paths[2])

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

    # ax.legend(loc='lower left', ncol=2)
    ax.legend(loc='lower left', ncol=1)

    ax.set_xticks([d * np.pi / 180 for d in [-20, -10, 0, 10, 20]])
    ax.set_thetalim(-np.pi / 8, np.pi / 8)

    return fig


def soft_robot_xy(in_paths: List[pathlib.Path]) -> plt.Figure:
    """Save soft robot xy timeseries plot."""
    edmd = _open_all_pickles(in_paths[0])
    srconst = _open_all_pickles(in_paths[1])
    hinf = _open_all_pickles(in_paths[2])
    hinfw = _open_all_pickles(in_paths[3])

    series = 'timeseries_15.0'

    t_step = 1 / edmd['bode']['f_samp']
    n_start = 300
    n_stop = None
    # n_start = 0
    # n_stop = 300
    t = np.arange(n_start, n_stop) * t_step

    n_x = edmd[series]['X_prediction'].shape[1] - 1

    fig, ax = plt.subplots(constrained_layout=True, sharex=True)

    ax.plot(
        edmd[series]['X_validation'][n_start:n_stop, 1],
        edmd[series]['X_validation'][n_start:n_stop, 2],
        '--k',
        label='Ground truth',
    )

    ax.plot(
        edmd[series]['X_prediction'][n_start:n_stop, 1],
        edmd[series]['X_prediction'][n_start:n_stop, 2],
        label='Extended DMD',
        color=C['edmd'],
    )
    ax.plot(
        srconst[series]['X_prediction'][n_start:n_stop, 1],
        srconst[series]['X_prediction'][n_start:n_stop, 2],
        label='A.S. constraint',
        color=C['srconst'],
    )
    ax.plot(
        hinf[series]['X_prediction'][n_start:n_stop, 1],
        hinf[series]['X_prediction'][n_start:n_stop, 2],
        label=f'{HINF} regularizer',
        color=C['hinf'],
    )


def soft_robot_time(in_paths: List[pathlib.Path]) -> plt.Figure:
    """Save soft robot timeseries plot."""
    edmd = _open_all_pickles(in_paths[0])
    srconst = _open_all_pickles(in_paths[1])
    hinf = _open_all_pickles(in_paths[2])
    hinfw = _open_all_pickles(in_paths[3])

    series = 'timeseries_15.0'

    t_step = 1 / edmd['bode']['f_samp']
    n_t = edmd[series]['X_validation'].shape[0]
    t = np.arange(n_t) * t_step

    n_x = edmd[series]['X_prediction'].shape[1] - 1

    fig, ax = plt.subplots(5, 1, constrained_layout=True, sharex=True)

    for i in range(ax.size):
        ax[i].plot(
            t,
            edmd[series]['X_validation'][:n_t, i + 1],
            '--',
            color=C['u1'],
            label='Ground truth',
        )

    for i in range(n_x):
        ax[i].plot(
            t,
            edmd[series]['X_prediction'][:n_t, i + 1],
            label='Extended DMD',
            color=C['edmd'],
        )
        ax[i].plot(
            t,
            srconst[series]['X_prediction'][:n_t, i + 1],
            label='A.S. constraint',
            color=C['srconst'],
        )
        ax[i].plot(
            t,
            hinf[series]['X_prediction'][:n_t, i + 1],
            label=f'{HINF} regularizer',
            color=C['hinf'],
        )
        # ax[i].plot(
        #     t,
        #     hinfw[series]['X_prediction'][:n_t, i + 1],
        #     label='hinfw',
        # )

    ax[-1].set_xlabel(r'$t$ (s)')

    ax[0].set_ylabel(r'$x_1(t)$ (cm)')
    ax[1].set_ylabel(r'$x_2(t)$ (cm)')
    ax[2].set_ylabel(r'$u_1(t)$ (V)')
    ax[3].set_ylabel(r'$u_2(t)$ (V)')
    ax[4].set_ylabel(r'$u_3(t)$ (V)')

    # ax[0].legend(loc='upper left')
    fig.legend(
        ax[0].get_lines(),
        [
            'Ground truth',
            'Extended DMD',
            'A.S. constraint',
            f'{HINF} regularizer',
        ],
        loc='lower right',
    )
    fig.align_labels()

    return fig


def soft_robot_error(in_paths: List[pathlib.Path]) -> plt.Figure:
    """Save soft robot timeseries plot."""
    edmd = _open_all_pickles(in_paths[0])
    srconst = _open_all_pickles(in_paths[1])
    hinf = _open_all_pickles(in_paths[2])
    hinfw = _open_all_pickles(in_paths[3])

    series = 'timeseries_15.0'

    t_step = 1 / edmd['bode']['f_samp']
    n_t = edmd[series]['X_validation'].shape[0]
    t = np.arange(n_t) * t_step

    n_x = edmd[series]['X_prediction'].shape[1] - 1

    fig, ax = plt.subplots(3, 1, constrained_layout=True, sharex=True)

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

    # ax[1].legend(loc='lower left', ncol=3, columnspacing=1.5)
    # ax[2].legend(loc='lower left', ncol=3, columnspacing=1.5)

    # ax[1].legend(handles = [l1,l2,l3] , labels=['A', 'B', 'C'],loc='upper center',
    #              bbox_to_anchor=(0.5, -0.2),fancybox=False, shadow=False, ncol=3)

    # ax2.legend(
    #     ax1.get_lines() + ax2.get_lines(),
    #     [
    #         f'{HINF} regularizer',
    #         f'Weighted {HINF} reg.',
    #         r'Weight',
    #     ],
    #     loc='upper right',
    #     # handlelength=1,
    # )

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

    return fig


def soft_robot_eig1(in_paths: List[pathlib.Path]) -> plt.Figure:
    """Save soft robot eigenvalue plot."""
    edmd = _open_all_pickles(in_paths[0])
    srconst = _open_all_pickles(in_paths[1])
    hinf = _open_all_pickles(in_paths[2])
    hinfw = _open_all_pickles(in_paths[3])

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

    ax.scatter(
        np.angle(edmd['eigenvalues']['eigv']),
        np.absolute(edmd['eigenvalues']['eigv']),
        color=C['edmd'],
        marker='x',
        s=80,
        linewidth=1.5,
        label=r'Extended DMD',
    )
    ax.scatter(
        np.angle(srconst['eigenvalues']['eigv']),
        np.absolute(srconst['eigenvalues']['eigv']),
        color=C['srconst'],
        facecolors='none',
        marker='o',
        s=80,
        linewidth=1.5,
        label=r'A.S. constraint',
    )
    ax.scatter(
        np.angle(hinf['eigenvalues']['eigv']),
        np.absolute(hinf['eigenvalues']['eigv']),
        color=C['hinf'],
        marker='+',
        s=80,
        linewidth=1.5,
        label=f'{HINF} regularizer',
    )

    ax.legend(loc='lower right', ncol=1)

    ax.set_rlim(0, 2.5)
    ax.set_yticks([0, 0.5, 1, 1.5, 2, 2.5])

    ax.set_xlabel(r'$\mathrm{Re}\{\lambda_i\}$')
    ax.set_ylabel(r'$\mathrm{Im}\{\lambda_i\}$', labelpad=25)

    return fig


def soft_robot_eig(in_paths: List[pathlib.Path]) -> plt.Figure:
    """Save soft robot eigenvalue plot."""
    edmd = _open_all_pickles(in_paths[0])
    srconst = _open_all_pickles(in_paths[1])
    hinf = _open_all_pickles(in_paths[2])
    hinfw = _open_all_pickles(in_paths[3])

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

    return fig


def soft_robot_bode(in_paths: List[pathlib.Path]) -> plt.Figure:
    """Save soft robot bode plot."""
    edmd = _open_all_pickles(in_paths[0])
    srconst = _open_all_pickles(in_paths[1])
    hinf = _open_all_pickles(in_paths[2])
    hinfw = _open_all_pickles(in_paths[3])

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
    return fig


def soft_robot_svd(in_paths: List[pathlib.Path]) -> plt.Figure:
    """Save soft robot SVD plot."""
    edmd = _open_all_pickles(in_paths[0])
    srconst = _open_all_pickles(in_paths[1])
    hinf = _open_all_pickles(in_paths[2])
    hinfw = _open_all_pickles(in_paths[3])

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

    return fig


def soft_robot_weights(in_paths: List[pathlib.Path]) -> plt.Figure:
    """Save soft robot bode weights."""
    hinf = _open_all_pickles(in_paths[0])
    hinfw = _open_all_pickles(in_paths[1])

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

    # b1 = 17
    # b2 = -5
    # n = 10
    b1 = 14
    b2 = -4
    n = 16
    ax1.set_ylim(b1, b1 + n)
    ax2.set_ylim(b2, b2 + n)
    loc1 = matplotlib.ticker.LinearLocator(numticks=((n // 2) + 1))
    loc2 = matplotlib.ticker.LinearLocator(numticks=((n // 2) + 1))
    ax1.yaxis.set_major_locator(loc1)
    ax2.yaxis.set_major_locator(loc2)

    # ax2.legend(
    #     ax1.get_lines() + ax2.get_lines(),
    #     [
    #         f'{HINF} regularizer',
    #         f'Weighted {HINF} reg.',
    #         r'Weight',
    #     ],
    #     loc='upper right',
    #     # handlelength=1,
    # )

    return fig


def soft_robot_bar_by_experiment(in_paths: List[pathlib.Path]) -> plt.Figure:
    """Save soft robot bar chart grouped by experiment."""
    errors = _calc_soft_robot_rms(in_paths)

    fig, ax = plt.subplots(constrained_layout=True)
    errors.plot(
        kind='bar',
        ax=ax,
        rot=0,
        edgecolor='w',
        color=[
            C['edmd'],
            C['srconst'],
            C['hinf'],
            C['hinfw'],
        ],
    )
    ax.set_xticklabels([r'\#1', r'\#2', r'\#3', r'\#4'])
    ax.set_xlabel('Validation set')
    ax.set_ylabel('RMS Euclidean error (cm)')
    ax.grid(axis='x')

    return fig


def soft_robot_scatter_by_method(
    in_paths: List[pathlib.Path],
    headers: List[str] = None,
    colors: List[Tuple[float, float, float]] = None,
    peak: float = None,
) -> plt.Figure:
    """Save soft robot bar chart grouped by method."""
    errors = _calc_soft_robot_rms(in_paths, headers)
    means = errors.mean()
    std = errors.std()

    fig, ax = plt.subplots(constrained_layout=True)

    c = ([C['edmd'], C['srconst'], C['hinf'], C['hinfw']]
         if colors is None else colors)
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

    scatter_style = {
        's': 50 * 1.5,
        'edgecolors': 'w',
        'linewidth': 0.25 * 1.5,
        'zorder': 2,
    }
    ax.scatter(
        x=xm,
        y=errors.iloc[0, :],
        c=c,
        marker='o',
        **scatter_style,
    )
    ax.scatter(
        x=xm,
        y=errors.iloc[1, :],
        c=c,
        marker='s',
        **scatter_style,
    )
    ax.scatter(
        x=xm,
        y=errors.iloc[2, :],
        c=c,
        marker='D',
        **scatter_style,
    )
    ax.scatter(
        x=xm,
        y=errors.iloc[3, :],
        c=c,
        marker='P',
        **scatter_style,
    )

    ax.scatter(
        x=-1,
        y=-1,
        c='k',
        marker='o',
        label=r'Valid. ep. \#1',
        **scatter_style,
    )
    ax.scatter(
        x=-1,
        y=-1,
        c='k',
        marker='s',
        label=r'Valid. ep. \#2',
        **scatter_style,
    )
    ax.scatter(
        x=-1,
        y=-1,
        c='k',
        marker='D',
        label=r'Valid. ep. \#3',
        **scatter_style,
    )
    ax.scatter(
        x=-1,
        y=-1,
        c='k',
        marker='P',
        label=r'Valid. ep. \#4',
        **scatter_style,
    )

    ax.set_xlabel('Regression method')
    ax.set_ylabel('RMS Euclidean error (cm)')
    ax.grid(axis='x')
    if peak is None:
        peak = 1.4
    ax.set_ylim(0, peak)
    # ax.set_yticks([0.1 * i for i in range(3, 14)])
    ax.set_xticks(x)
    ax.set_xticklabels([errors.columns[i] for i in range(len(x))])

    ax.legend(loc='upper right')

    ax.set_xlim(-0.5, 3.5)

    return fig


def soft_robot_bar_by_method(in_paths: List[pathlib.Path]) -> plt.Figure:
    """Save soft robot bar chart grouped by method."""
    errors = _calc_soft_robot_rms(in_paths)

    fig, ax = plt.subplots(constrained_layout=True)
    means = errors.mean()
    mins = means - errors.min()
    maxs = errors.max() - means
    means.plot.bar(
        yerr=[mins, maxs],
        ax=ax,
        capsize=4,
        color=[
            C['edmd'],
            C['srconst'],
            C['hinf'],
            C['hinfw'],
        ],
        rot=0,
        linewidth=1.5,
    )
    ax.set_xlabel('Regression method')
    ax.set_ylabel('RMS Euclidean error (cm)')
    ax.grid(axis='x')

    return fig


def soft_robot_scatter_dmdc(in_paths: List[pathlib.Path]) -> plt.Figure:
    """Save soft robot bar chart grouped by method."""
    return soft_robot_scatter_by_method(
        in_paths,
        headers=[
            'EDMD,\nA.S. constr.',
            'DMDc,\nA.S. constr.',
            f'EDMD,\n{HINF} reg.',
            f'DMDc,\n{HINF} reg.',
        ],
        colors=[C['srconst'], C['srconst_dmdc'], C['hinf'], C['hinf_dmdc']],
        peak=2,
    )


def soft_robot_scatter_modified(in_paths: List[pathlib.Path]) -> plt.Figure:
    """Save soft robot bar chart grouped by method."""
    return soft_robot_scatter_by_method(in_paths, headers=['A', 'B', 'C', 'D'])


def soft_robot_dmdc_svd(in_paths: List[pathlib.Path]) -> plt.Figure:
    """Save soft robot DMDc SVD plot."""
    srconst = _open_all_pickles(in_paths[0])
    srconst_dmdc = _open_all_pickles(in_paths[1])
    hinf = _open_all_pickles(in_paths[2])
    hinf_dmdc = _open_all_pickles(in_paths[3])

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

    return fig


def soft_robot_dmdc_bode(in_paths: List[pathlib.Path]) -> plt.Figure:
    """Save soft robot DMDc bode plot."""
    srconst = _open_all_pickles(in_paths[0])
    srconst_dmdc = _open_all_pickles(in_paths[1])
    hinf = _open_all_pickles(in_paths[2])
    hinf_dmdc = _open_all_pickles(in_paths[3])

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

    return fig


def soft_robot_ram(in_paths: List[pathlib.Path]) -> plt.Figure:
    """Save soft robot performance plot."""
    srconst = _load_dat(in_paths[0])
    srconst_dmdc = _load_dat(in_paths[1])
    hinf = _load_dat(in_paths[2])
    hinf_dmdc = _load_dat(in_paths[3])

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

    return fig


def soft_robot_exec(in_paths: List[pathlib.Path]) -> plt.Figure:
    """Save soft robot performance plot."""
    srconst = _load_dat(in_paths[0])
    srconst_dmdc = _load_dat(in_paths[1])
    hinf = _load_dat(in_paths[2])
    hinf_dmdc = _load_dat(in_paths[3])

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

    return fig


def _load_dat(path: pathlib.Path) -> Tuple[float, float]:
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


def _calc_soft_robot_rms(in_paths: List[pathlib.Path],
                         headers: List[str] = None) -> pandas.DataFrame:
    """Form a DataFrame with RMS errors for soft robot."""
    edmd = _open_all_pickles(in_paths[0])
    srconst = _open_all_pickles(in_paths[1])
    hinf = _open_all_pickles(in_paths[2])
    hinfw = _open_all_pickles(in_paths[3])

    def calc_rms(loaded_pickle):
        datasets = [f'timeseries_{n}.0' for n in ['13', '14', '15', '16']]
        rmses = []
        for ds in datasets:
            pred = loaded_pickle[ds]['X_prediction'][:, 1:]
            vald = loaded_pickle[ds]['X_validation'][:, 1:(pred.shape[1] + 1)]
            err = np.linalg.norm(vald - pred, axis=1)
            rmse = np.sqrt(np.mean(err**2))
            rmses.append(rmse)
        return rmses

    if headers is None:
        headers = [
            'Extended DMD',
            'A.S. constraint',
            f'{HINF} regularizer',
            f'Weighted {HINF} reg.',
        ]

    errors = pandas.DataFrame({
        headers[0]: calc_rms(edmd),
        headers[1]: calc_rms(srconst),
        headers[2]: calc_rms(hinf),
        headers[3]: calc_rms(hinfw),
    })

    return errors


def _open_all_pickles(path: pathlib.Path) -> Dict[str, Any]:
    """Open all pickles in directory and return dict of data."""
    pickles = list(path.glob('*.pickle'))
    keys = [p.stem for p in pickles]
    opened_pickles = {}
    for k, p in zip(keys, pickles):
        with open(p, 'rb') as f:
            opened_pickles[k] = pickle.load(f)
    return opened_pickles


if __name__ == '__main__':
    main()
