"""Run a Koopman experiment."""

import datetime
import logging
import os
import pathlib
import pickle
import subprocess
import time
from typing import Any, Dict, Optional, Tuple

import hydra
import numpy as np
import omegaconf
import pykoop
import pykoop.lmi_regressors
import sklearn.preprocessing
from matplotlib import pyplot as plt
from scipy import linalg, signal


@hydra.main(config_path='config', config_name='config')
def main(config: omegaconf.DictConfig) -> None:
    """Run a Koopman experiment.

    Dataset must contain: ``n_inputs``, ``episode_feature``, ``t_step``, ``X``,
    ``training_episodes``, and ``validation_episodes``.
    """
    # Keep track of time
    start_time = time.monotonic()
    # Configure matplotlib
    plt.rc('figure', figsize=(16, 9))
    plt.rc('lines', linewidth=2)
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    # Get working directory
    wd = pathlib.Path(os.getcwd())
    # Dict where all relevant results will be saved:
    res: Dict[str, Dict[str, Any]] = {}
    # Load dataset from config
    original_wd = pathlib.Path(hydra.utils.get_original_cwd())
    dataset_path = original_wd.joinpath(config.dataset)
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    # Instantiate lifting functions from config
    lifting_functions: Optional[pykoop.KoopmanLiftingFn]
    if config.lifting_functions.lifting_functions:
        lifting_functions = []
        for key, lf in config.lifting_functions.lifting_functions:
            lifting_functions.append((key, hydra.utils.instantiate(lf)))
    else:
        lifting_functions = None
    # Instantiate regressor from config
    regressor = hydra.utils.instantiate(config.regressor.regressor)
    if 't_step' in regressor.get_params().keys():
        regressor.set_params(t_step=dataset['t_step'])
    # Instantiate Koopman pipeline
    kp = pykoop.KoopmanPipeline(
        lifting_functions=lifting_functions,
        regressor=regressor,
    )
    # Log config contents
    logging.info(f'Config: {config}')
    # Figure out smallest number of training and validation timesteps
    n_steps_training, n_steps_validation = calc_n_steps(dataset)
    # Split training and validation data
    X_training, X_validation = split_training_validation(
        dataset, n_steps_training)
    # Fit pipeline (wrap in Memory Profiler ``profile`` if required)
    if not config.profile:
        kp.fit(X_training,
               n_inputs=dataset['n_inputs'],
               episode_feature=dataset['episode_feature'])
    else:
        # ``@profile`` decorator is defined by ``mprof run --python ...``
        profile(kp.fit)(X_training,
                        n_inputs=dataset['n_inputs'],
                        episode_feature=dataset['episode_feature'])
    # Save fit estimator
    with open(wd.joinpath('estimator.pickle'), 'wb') as f:
        pickle.dump(kp, f)
    # Plot weights if present
    if (hasattr(kp.regressor_, 'ss_ct_') and hasattr(kp.regressor_, 'ss_dt_')):
        plot_weights(
            'weights',
            wd,
            res,
            kp.regressor_.ss_ct_,
            kp.regressor_.ss_dt_,
        )
    # Plot validation set prediction and errors
    episodes = pykoop.split_episodes(
        X_validation, episode_feature=dataset['episode_feature'])
    for (i, X_i) in episodes:
        X_i_with_ep = np.hstack((
            i * np.ones((X_i.shape[0], 1)),
            X_i,
        ))
        plot_timeseries(f'timeseries_{i}', wd, res, X_i_with_ep, [kp])
        plot_error(f'error_{i}', wd, res, X_i_with_ep, [kp])
    # Plot other interesting estimator propreties
    plot_eigenvalues('eigenvalues', wd, res, [kp])
    plot_matshow('matshow', wd, res, kp)
    plot_mimo_bode('bode', wd, res, kp, dataset['t_step'])
    plot_convergence('convergence', wd, res, kp)
    # Save pickle of results
    with open(wd.joinpath('run_experiment.pickle'), 'wb') as f:
        pickle.dump(res, f)
    # End timer
    end_time = time.monotonic()
    execution_time = end_time - start_time
    # Format execution time nicely
    formatted_execution_time = datetime.timedelta(seconds=execution_time)
    # Log execution time
    logging.info(f'Execution time: {formatted_execution_time}')
    # Send push notification if ``ntfy`` is installed and configured.
    if config.notify:
        cfg = hydra.core.hydra_config.HydraConfig.get().job.override_dirname
        status = f'Config: {cfg}\nExecution time: {formatted_execution_time}'
        try:
            subprocess.call(('ntfy', '--title', 'Job done', 'send', status))
        except Exception:
            logging.warning('To enable push notifications, install `ntfy` '
                            'from: https://github.com/dschep/ntfy')


def calc_n_steps(dataset: Dict) -> Tuple[int, int]:
    """Find smallest number of training and validation timesteps."""
    sizes_training = []
    sizes_validation = []
    # Split dataset into episodes
    episodes = pykoop.split_episodes(
        dataset['X'], episode_feature=dataset['episode_feature'])
    # Iterate over episodes, logging shape
    for (i, X_i) in episodes:
        if i in dataset['validation_episodes']:
            # Episode is in validation set
            sizes_validation.append(X_i.shape[0])
        else:
            # Episode is in training set
            # If there's no episode feature, everything will fall here
            sizes_training.append(X_i.shape[0])
    # Calculate minimum sizes
    n_steps_training = np.min(sizes_training)
    n_steps_validation = np.min(sizes_validation)
    if n_steps_validation > n_steps_training:
        logging.warning('More validation timesteps than training.')
    return (n_steps_training, n_steps_validation)


def split_training_validation(
        dataset: Dict, n_steps_training: int) -> Tuple[np.ndarray, np.ndarray]:
    """Split training and validation data."""
    if dataset['episode_feature']:
        # If there's an episode feature, split the episodes using that
        training_idx = np.where(
            np.in1d(dataset['X'][:, 0], dataset['training_episodes']))[0]
        validation_idx = np.where(
            np.in1d(dataset['X'][:, 0], dataset['validation_episodes']))[0]
        X_training = dataset['X'][training_idx, :]
        X_validation = dataset['X'][validation_idx, :]
    else:
        # If there's no episode feature, split the data in half
        X_training = dataset['X'][:(n_steps_training // 2), :]
        X_validation = dataset['X'][(n_steps_training // 2):, :]
    return (X_training, X_validation)


def plot_timeseries(path, wd, res, X_validation, estimators, labels=None):
    """Plot timeseries of states."""
    # Create figure
    fig = plt.figure(constrained_layout=True)
    # Fill in labels if missing
    if labels is None:
        labels = []
        for est in estimators:
            labels.append(type(est.regressor_).__name__)
    # Create gridspec
    n_state = estimators[0].n_states_in_
    n_input = estimators[0].n_inputs_in_
    n_method = len(estimators)
    gs = fig.add_gridspec(n_state + n_input, n_method)
    ax = np.empty((n_state + n_input, n_method), dtype=object)
    for i in range(n_state + n_input):
        for j in range(n_method):
            ax[i, j] = fig.add_subplot(gs[i, j])
            ax[i, j].grid(True, linestyle='--')
            ax[i, j].set_xlabel(r'$k$')
            if i < n_state:
                ax[i, j].set_ylabel(rf'$x_{i}[k]$')
            else:
                ax[i, j].set_ylabel(rf'$u_{i - n_state}[k]$')
    # Predict and plot
    for (j, est, lab) in zip(range(n_method), estimators, labels):
        try:
            X_prediction = est.predict_multistep(X_validation)
            scorer = pykoop.KoopmanPipeline.make_scorer()
            score = scorer(est, X_validation)
        except Exception as e:
            logging.warning(e)
            score = np.nan
        res[path] = {
            'X_prediction': X_prediction,
            'X_validation': X_validation,
        }
        X_prediction = X_prediction[:, 1:]
        X_validation = X_validation[:, 1:]
        for i in range(n_state + n_input):
            if i < n_state:
                ax[i, j].plot(X_validation[:, i], label='True state')
                ax[i, j].plot(X_prediction[:, i], label='Predicted state')
            else:
                ax[i, j].plot(X_validation[:, i])
            ax[0, j].set_title(lab + f' MSE: {-1 * score}')
            ax[0, j].legend(loc='lower right')
    # Save figure
    fig.savefig(wd.joinpath(f'{path}.png'))


def plot_error(path, wd, res, X_validation, estimators, labels=None):
    """Plot timeseries of error."""
    # Create figure
    fig = plt.figure(constrained_layout=True)
    # Fill in labels if missing
    if labels is None:
        labels = []
        for est in estimators:
            labels.append(type(est.regressor_).__name__)

    # Create gridspec
    n_state = estimators[0].n_states_in_
    n_method = len(estimators)
    gs = fig.add_gridspec(n_state, n_method)
    ax = np.empty((n_state, n_method), dtype=object)
    for i in range(n_state):
        for j in range(n_method):
            ax[i, j] = fig.add_subplot(gs[i, j])
            ax[i, j].grid(True, linestyle='--')
            ax[i, j].set_xlabel(r'$k$')
            ax[i, j].set_ylabel(rf'$e_{i}[k]$')

    # Predict and plot
    for (j, est, lab) in zip(range(n_method), estimators, labels):
        try:
            X_prediction = est.predict_multistep(X_validation)
            scorer = pykoop.KoopmanPipeline.make_scorer()
            score = scorer(est, X_validation)
        except Exception as e:
            logging.warning(e)
            score = np.nan
        X_prediction = X_prediction[:, 1:]
        X_validation = X_validation[:, 1:]
        for i in range(n_state):
            ax[0, j].set_title(lab + f' MSE: {-1 * score}')
            ax[i, j].plot(X_validation[:, i] - X_prediction[:, i])
    # Save figure
    fig.savefig(wd.joinpath(f'{path}.png'))


def plot_weights(path, wd, res, ss_ct, ss_dt):
    """Plot Hinf weights."""
    fig, ax = plt.subplots(1, 2, constrained_layout=True)
    ax[0].grid(True, linestyle='--')
    ax[1].grid(True, linestyle='--')
    # Continuous time
    w_ct, H_ct = signal.freqresp(ss_ct)
    mag_ct = 20 * np.log10(np.abs(H_ct))
    ax[0].semilogx(w_ct, mag_ct)
    ax[0].set_xlabel('Frequency [rad/s]')
    ax[0].set_ylabel('Magnitude [dB]')
    ax[0].set_title('Continuous-time weight')
    # Discrete time
    w_dt, H_dt = signal.dfreqresp(ss_dt)
    mag_dt = np.abs(H_dt)
    mag_dt_db = 20 * np.log10(mag_dt)
    ax[1].plot(w_dt, mag_dt, color='C0')
    ax[1].set_xlabel('Frequency [rad/sample]')
    ax[1].set_ylabel('Magnitude', color='C0')
    ax[1].tick_params(axis='y', labelcolor='C0')
    ax[1].set_title('Discrete-time weight')
    ax2 = ax[1].twinx()
    ax2.plot(w_dt, mag_dt_db, color='C1')
    ax2.set_ylabel('Magnitude [dB]', color='C1')
    ax2.tick_params(axis='y', labelcolor='C1')
    res[path] = {
        'w_ct': w_ct,
        'H_ct': H_ct,
        'mag_ct': mag_ct,
        'w_dt': w_dt,
        'H_dt': H_dt,
        'mag_dt': mag_dt,
        'mag_dt_db': mag_dt_db,
    }
    # Save figure
    fig.savefig(wd.joinpath(f'{path}.png'))


def plot_eigenvalues(path, wd, res, estimators, labels=None):
    """Plot eigendecomposition."""
    # Create figure
    fig = plt.figure(constrained_layout=True)

    def plt_uc(ax):
        th = np.linspace(0, 2 * np.pi)
        ax.plot(th, np.ones(th.shape), '--k')

    def plt_eig(A, ax, label='', marker='x'):
        """Eigenvalue plotting helper function."""
        eigv = linalg.eig(A)[0]
        eigv_mag = np.abs(eigv)
        idx = eigv_mag.argsort()[::-1]
        eigv_sort = eigv[idx]
        ax.scatter(np.angle(eigv_sort),
                   np.absolute(eigv_sort),
                   marker=marker,
                   label=label,
                   cmap='viridis')

    def plt_eig_mag(A, ax, label='', marker='x'):
        """Eigenvalue plotting helper function."""
        eigv = linalg.eig(A)[0]
        eigv_mag = np.absolute(eigv)
        idx = eigv_mag.argsort()[::-1]
        ax.plot(eigv_mag[idx], marker=marker, label=label)
        res[path] = {
            'eigv': eigv,
            'eigv_mat': eigv_mag[idx],
        }

    # Fill in labels if missing
    if labels is None:
        labels = []
        for est in estimators:
            labels.append(type(est.regressor_).__name__)

    # Create gridspec
    n_plt = len(estimators)
    gs = fig.add_gridspec(2, n_plt)
    ax = np.empty((2, n_plt), dtype=object)
    for i in range(n_plt):
        ax[0, i] = fig.add_subplot(gs[0, i], projection='polar')
        ax[0, i].set_xlabel(r'$\mathrm{Re}(\lambda)$')
        ax[0, i].set_ylabel(r'$\mathrm{Im}(\lambda)$', labelpad=30)
        ax[0, i].set_rmax(10)
        ax[0, i].grid(True, linestyle='--')
    for i in range(n_plt):
        ax[1, i] = fig.add_subplot(gs[1, i])
        ax[1, i].set_xlabel(r'$i$')
        ax[1, i].set_ylabel(r'$\|\lambda_i\|$')
        ax[1, i].grid(True, linestyle='--')

    # Plot contents
    for (i, est, lab) in zip(range(n_plt), estimators, labels):
        U = est.regressor_.coef_.T
        A = U[:, :U.shape[0]]
        ax[0, i].set_title(lab)
        plt_uc(ax[0, i])
        plt_eig(A, ax[0, i], label=lab)
        plt_eig_mag(A, ax[1, i], label=lab)
    # Save figure
    fig.savefig(wd.joinpath(f'{path}.png'))


def plot_matshow(path, wd, res, estimator, label=None):
    """Plot matrix in an image."""
    # Fill in labels if missing
    if label is None:
        label = type(estimator.regressor_).__name__
    # Get Koopman matrix
    U = estimator.regressor_.coef_.T
    p_theta, p = U.shape
    # Plot Koopman matrix and dividing line between ``A`` and ``B``.
    fig, ax = plt.subplots(constrained_layout=True)
    mag = np.max(np.abs(U))
    im = ax.matshow(U, vmin=-mag, vmax=mag, cmap='seismic')
    ax.vlines(p_theta - 0.5, -0.5, p_theta - 0.5, color='green')
    ax.set_title(label)
    fig.colorbar(im, ax=ax)
    # Save figure
    res[path] = {
        'U': U,
    }
    fig.savefig(wd.joinpath(f'{path}.png'))


def plot_mimo_bode(path, wd, res, estimator, t_step, label=None):
    """Plot MIMO Bode plot."""
    # Fill in labels if missing
    if label is None:
        label = type(estimator.regressor_).__name__
    # Get Koopman matrix
    U = estimator.regressor_.coef_.T
    # Get ``A``, ``B``, and ``C``.
    A = U[:, :U.shape[0]]
    B = U[:, U.shape[0]:]
    C = np.eye(U.shape[0])

    def sigma_bar_G(f):
        """Maximum singular value of transfer matrix at a frequency."""
        z = np.exp(1j * 2 * np.pi * f * t_step)
        G = C @ linalg.solve((np.diag([z] * A.shape[0]) - A), B)
        sigma_bar_G = linalg.svdvals(G)[0]
        return sigma_bar_G

    # Compute magnitudes
    f_samp = 1 / t_step
    f_plot = np.linspace(0, f_samp / 2, 1000)
    bode = []
    for f in f_plot:
        bode.append(sigma_bar_G(f))
    mag = np.array(bode)
    mag_db = 20 * np.log10(mag)
    # Construct Bode plot
    fig, ax = plt.subplots(constrained_layout=True)
    ax.grid(True, linestyle='--')
    ax.plot(f_plot, mag, color='C0')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Maximum singular value of G[z]', color='C0')
    ax.tick_params(axis='y', labelcolor='C0')
    ax.set_title(label)
    ax2 = ax.twinx()
    ax2.plot(f_plot, mag_db, color='C1')
    ax2.set_ylabel('Maximum singular value of G[z] (dB)', color='C1')
    ax2.tick_params(axis='y', labelcolor='C1')
    # Save figure
    res[path] = {
        'f_samp': f_samp,
        'f_plot': f_plot,
        'mag': mag,
        'mag_db': mag_db,
    }
    fig.savefig(wd.joinpath(f'{path}.png'))


def plot_convergence(path, wd, res, estimator, label=None):
    """Plot convergence if applicable."""
    # Check for objective log
    obj_log = None
    if hasattr(estimator.regressor_, 'objective_log_'):
        obj_log = estimator.regressor_.objective_log_
    elif hasattr(estimator.regressor_, 'hinf_regressor_'):
        obj_log = estimator.regressor_.hinf_regressor_.objective_log_
    # Plot objective log
    if obj_log is not None:
        # Fill in labels if missing
        if label is None:
            label = type(estimator.regressor_).__name__
        # Get objective
        obj = np.array(obj_log)
        # Plot objective
        fig, ax = plt.subplots(constrained_layout=True)
        ax.grid(True, linestyle='--')
        ax.plot(obj)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Objective function value')
        # Save figure
        res[path] = {
            'obj': obj,
        }
        fig.savefig(wd.joinpath(f'{path}.png'))


if __name__ == '__main__':
    main()
