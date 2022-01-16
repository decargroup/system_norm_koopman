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

# --------------------------------------------------------------------------- #
# Main job
# --------------------------------------------------------------------------- #


@hydra.main(config_path='config', config_name='config')
def main(config: omegaconf.DictConfig) -> None:
    """Run a Koopman experiment.

    Dataset must contain: ``n_inputs``, ``episode_feature``, ``t_step``, ``X``,
    ``training_episodes``, and ``validation_episodes``.
    """
    # Keep track of time
    start_time = time.monotonic()
    # Configure Matplotlib
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
    # Set regressor timestep from dataset after instantiation if required
    if 't_step' in regressor.get_params().keys():
        regressor.set_params(t_step=dataset['t_step'])
    # Instantiate Koopman pipeline
    kp = pykoop.KoopmanPipeline(
        lifting_functions=lifting_functions,
        regressor=regressor,
    )
    # Log contents of config
    logging.info(f'Config: {config}')
    # Find smallest number of training and validation timesteps
    n_steps_training, n_steps_validation = calc_n_steps(dataset)
    # Split training and validation data
    X_training, X_validation = split_training_validation(
        dataset,
        n_steps_training,
    )
    # Fit pipeline (wrap in Memory Profiler ``profile`` if required)
    if not config.profile:
        # Fit pipeline normally
        kp.fit(
            X_training,
            n_inputs=dataset['n_inputs'],
            episode_feature=dataset['episode_feature'],
        )
    else:
        # Fit pipeline while profiling fit function with Memory Profiler
        # ``@profile`` decorator is defined by ``mprof run --python ...``
        profile(kp.fit)(
            X_training,
            n_inputs=dataset['n_inputs'],
            episode_feature=dataset['episode_feature'],
        )
    # Save fit estimator to pickle
    with open(wd.joinpath('estimator.pickle'), 'wb') as f:
        pickle.dump(kp, f)
    # Plot weights if present
    if (hasattr(kp.regressor_, 'ss_ct_') and hasattr(kp.regressor_, 'ss_dt_')):
        # Continuous time response
        w_ct, H_ct = signal.freqresp(kp.regressor_.ss_ct_)
        mag_ct = 20 * np.log10(np.abs(H_ct))
        # Discrete time response
        w_dt, H_dt = signal.dfreqresp(kp.regressor_.ss_dt_)
        mag_dt = np.abs(H_dt)
        mag_dt_db = 20 * np.log10(mag_dt)
        # Save weight results
        key = 'weights'
        res[key] = {
            'w_ct': w_ct,
            'H_ct': H_ct,
            'mag_ct': mag_ct,
            'w_dt': w_dt,
            'H_dt': H_dt,
            'mag_dt': mag_dt,
            'mag_dt_db': mag_dt_db,
        }
        # Plot weight results
        fig = plot_weights(w_ct, mag_ct, w_dt, mag_dt, mag_dt_db)
        fig.savefig(wd.joinpath(f'{key}.png'))
    # Split validation episodes using episode feature
    episodes = pykoop.split_episodes(
        X_validation,
        episode_feature=dataset['episode_feature'],
    )
    # Iterate over all validation episodes and plot/save them
    for (i, X_i) in episodes:
        # Re-add episode feature to current validation set
        X_validation_i = np.hstack((i * np.ones((X_i.shape[0], 1)), X_i))
        # Perform prediction with fit estimator
        X_prediction = kp.predict_multistep(X_validation_i)
        # Calculate score. If the prediction diverges, set the score to NaN
        try:
            scorer = pykoop.KoopmanPipeline.make_scorer()
            score = scorer(kp, X_validation_i)
        except Exception as e:
            logging.warning(e)
            score = np.nan
        # Save results in dict
        key = f'timeseries_{i}'
        res[key] = {
            'X_prediction': X_prediction,
            'X_validation': X_validation_i,
        }
        # Plot prediction and validation timeseries
        fig = plot_timeseries(X_validation_i, X_prediction, score)
        fig.savefig(wd.joinpath(f'{key}.png'))
        # Plot error
        key = f'error_{i}'
        fig = plot_error(X_validation_i, X_prediction, score)
        fig.savefig(wd.joinpath(f'{key}.png'))
    # Extract Koopman matrix
    U = kp.regressor_.coef_.T
    A = U[:, :U.shape[0]]
    B = U[:, U.shape[0]:]
    # Compute eigenvalues of Koopman matrix
    eigv = linalg.eig(A)[0]
    eigv_mag = np.absolute(eigv)
    idx = eigv_mag.argsort()[::-1]
    eigv_mag_sorted = eigv_mag[idx]
    # Save eigenvalues
    key = 'eigenvalues'
    res[key] = {
        'eigv': eigv,
        'eigv_mag': eigv_mag_sorted,
    }
    # Plot eigenvalues
    fig = plot_eigenvalues(eigv, eigv_mag_sorted)
    fig.savefig(wd.joinpath(f'{key}.png'))
    # Save Koopman matrix
    key = 'matshow'
    res[key] = {
        'U': U,
    }
    # Plot Koopman matrix
    fig = plot_matshow(U)
    fig.savefig(wd.joinpath(f'{key}.png'))
    # Compute MIMO frequency response of Koopman system
    C = np.eye(U.shape[0])
    f_samp = 1 / dataset['t_step']
    f_plot = np.linspace(0, f_samp / 2, 1000)
    bode = []
    for f in f_plot:
        bode.append(sigma_bar_G(f, dataset['t_step'], A, B, C))
    mag = np.array(bode)
    mag_db = 20 * np.log10(mag)
    # Save MIMO frequency response
    key = 'bode'
    res[key] = {
        'f_samp': f_samp,
        'f_plot': f_plot,
        'mag': mag,
        'mag_db': mag_db,
    }
    # Plot MIMO frequency response
    fig = plot_mimo_bode(f_plot, mag, mag_db)
    fig.savefig(wd.joinpath(f'{key}.png'))
    # If the regressor was a BMI solved through iteration, extract the
    # convergence information
    obj_log: Optional[np.ndarray] = None
    if hasattr(kp.regressor_, 'objective_log_'):
        obj_log = np.array(kp.regressor_.objective_log_)
    elif hasattr(kp.regressor_, 'hinf_regressor_'):
        # Special case for ``LmiHinfZpkMeta``
        obj_log = np.array(kp.regressor_.hinf_regressor_.objective_log_)
    # Save and plot the convergence information if present
    if obj_log is not None:
        key = 'convergence'
        res[key] = {
            'obj': obj_log,
        }
        fig = plot_convergence(obj_log)
        fig.savefig(wd.joinpath(f'{key}.png'))
    # Save pickle of all results
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
        # Form string to send in push notification
        cfg = hydra.core.hydra_config.HydraConfig.get().job.override_dirname
        status = f'Config: {cfg}\nExecution time: {formatted_execution_time}'
        try:
            subprocess.call(('ntfy', '--title', 'Job done', 'send', status))
        except Exception as e:
            logging.warning(e)
            logging.warning('To enable push notifications, install `ntfy` '
                            'from: https://github.com/dschep/ntfy')


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #


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


def sigma_bar_G(f: float, t_step: float, A: np.ndarray, B: np.ndarray,
                C: np.ndarray) -> np.ndarray:
    """Maximum singular value of transfer matrix at a frequency."""
    z = np.exp(1j * 2 * np.pi * f * t_step)
    G = C @ linalg.solve((np.diag([z] * A.shape[0]) - A), B)
    sigma_bar_G = linalg.svdvals(G)[0]
    return sigma_bar_G


# --------------------------------------------------------------------------- #
# Plotting functions
# --------------------------------------------------------------------------- #


def plot_timeseries(X_validation: np.ndarray, X_prediction: np.ndarray,
                    score: float) -> plt.Figure:
    # Compute state and input dimensions
    n_state = X_prediction.shape[1] - 1
    n_input = X_validation.shape[1] - n_state - 1
    # Ditch episode feature
    X_pred = X_prediction[:, 1:]
    X_vald = X_validation[:, 1:]
    fig, ax = plt.subplots(n_state + n_input, 1, constrained_layout=True)
    for i in range(n_state + n_input):
        ax[i].grid(True, linestyle='--')
        ax[i].set_xlabel(r'$k$')
        if i < n_state:
            ax[i].plot(X_vald[:, i], label='True state')
            ax[i].plot(X_pred[:, i], label='Predicted state')
            ax[i].set_ylabel(rf'$x_{i}[k]$')
        else:
            ax[i].plot(X_vald[:, i])
            ax[i].set_ylabel(rf'$u_{i - n_state}[k]$')
        ax[0].set_title(f' MSE: {-1 * score}')
        ax[0].legend(loc='lower right')
    return fig


def plot_error(X_validation: np.ndarray, X_prediction: np.ndarray,
               score: float) -> plt.Figure:
    # Compute state and input dimensions
    n_state = X_prediction.shape[1] - 1
    n_input = X_validation.shape[1] - n_state - 1
    # Ditch episode feature
    X_pred = X_prediction[:, 1:]
    X_vald = X_validation[:, 1:]
    fig, ax = plt.subplots(n_state + n_input, 1, constrained_layout=True)
    for i in range(n_state + n_input):
        ax[i].grid(True, linestyle='--')
        ax[i].set_xlabel(r'$k$')
        if i < n_state:
            ax[i].plot(X_vald[:, i] - X_pred[:, i], label='Prediction error')
            ax[i].set_ylabel(rf'$\Delta x_{i}[k]$')
        else:
            ax[i].plot(X_vald[:, i])
            ax[i].set_ylabel(rf'$u_{i - n_state}[k]$')
        ax[0].set_title(f' MSE: {-1 * score}')
        ax[0].legend(loc='lower right')
    return fig


def plot_weights(w_ct: np.ndarray, mag_ct: np.ndarray, w_dt: np.ndarray, mag_dt: np.ndarray, mag_dt_db: np.ndarray) -> plt.Figure:
    """Plot Hinf weights."""
    fig, ax = plt.subplots(1, 2, constrained_layout=True)
    ax[0].grid(True, linestyle='--')
    ax[0].semilogx(w_ct, mag_ct)
    ax[0].set_xlabel('Frequency [rad/s]')
    ax[0].set_ylabel('Magnitude [dB]')
    ax[0].set_title('Continuous-time weight')
    ax[1].grid(True, linestyle='--')
    ax[1].plot(w_dt, mag_dt, color='C0')
    ax[1].set_xlabel('Frequency [rad/sample]')
    ax[1].set_ylabel('Magnitude', color='C0')
    ax[1].tick_params(axis='y', labelcolor='C0')
    ax[1].set_title('Discrete-time weight')
    ax2 = ax[1].twinx()
    ax2.plot(w_dt, mag_dt_db, color='C1')
    ax2.set_ylabel('Magnitude [dB]', color='C1')
    ax2.tick_params(axis='y', labelcolor='C1')
    return fig


def plot_eigenvalues(eigv: np.ndarray, eigv_mag: np.ndarray) -> plt.Figure:
    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(2, 1)
    ax = np.empty((2, ), dtype=object)
    # Add polar plot
    ax[0] = fig.add_subplot(gs[0, 0], projection='polar')
    ax[0].set_xlabel(r'$\mathrm{Re}(\lambda)$')
    ax[0].set_ylabel(r'$\mathrm{Im}(\lambda)$', labelpad=30)
    ax[0].set_rmax(10)
    ax[0].grid(True, linestyle='--')
    # Add magnitude plot
    ax[1] = fig.add_subplot(gs[1, 0])
    ax[1].set_xlabel(r'$i$')
    ax[1].set_ylabel(r'$\|\lambda_i\|$')
    ax[1].grid(True, linestyle='--')
    # Plot polar plot
    th = np.linspace(0, 2 * np.pi)
    ax[0].plot(th, np.ones(th.shape), '--k')
    ax[0].scatter(np.angle(eigv), np.absolute(eigv), marker='x')
    # Plot magnitude plot
    ax[1].plot(eigv_mag, marker='x')
    return fig


def plot_matshow(U: np.ndarray) -> plt.Figure:
    p_theta, p = U.shape
    # Plot Koopman matrix and dividing line between ``A`` and ``B``.
    fig, ax = plt.subplots(constrained_layout=True)
    mag = np.max(np.abs(U))
    im = ax.matshow(U, vmin=-mag, vmax=mag, cmap='seismic')
    ax.vlines(p_theta - 0.5, -0.5, p_theta - 0.5, color='green')
    fig.colorbar(im, ax=ax)
    return fig


def plot_mimo_bode(f_plot: np.ndarray, mag: np.ndarray, mag_db: np.ndarray) -> plt.Figure:
    """Plot MIMO Bode plot."""
    fig, ax = plt.subplots(constrained_layout=True)
    ax.grid(True, linestyle='--')
    ax.plot(f_plot, mag, color='C0')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Maximum singular value of G[z]', color='C0')
    ax.tick_params(axis='y', labelcolor='C0')
    ax2 = ax.twinx()
    ax2.plot(f_plot, mag_db, color='C1')
    ax2.set_ylabel('Maximum singular value of G[z] (dB)', color='C1')
    ax2.tick_params(axis='y', labelcolor='C1')
    return fig


def plot_convergence(obj_log: np.ndarray) -> Figure:
    fig, ax = plt.subplots(constrained_layout=True)
    ax.grid(True, linestyle='--')
    ax.plot(obj_log)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Objective function value')
    return fig


if __name__ == '__main__':
    main()
