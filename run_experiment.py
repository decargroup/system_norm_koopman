"""Run a Koopman experiment."""

import datetime
import logging
import os
import pathlib
import pickle
import subprocess
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import hydra
import numpy as np
import omegaconf
import pandas
import pykoop
import pykoop.lmi_regressors
import sklearn.model_selection
import sklearn.preprocessing
import skopt
import skopt.callbacks
import skopt.plots
from matplotlib import pyplot as plt
from scipy import linalg, signal
from sklearn.experimental import enable_halving_search_cv


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
    plt.rc('axes', grid=True)
    plt.rc('grid', linestyle='--')

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Get working directory
    wd = pathlib.Path(os.getcwd())

    # Load data
    original_wd = pathlib.Path(hydra.utils.get_original_cwd())
    dataset_path = original_wd.joinpath(config.dataset)
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)

    # Instantiate lifting functions
    lifting_functions: Optional[pykoop.KoopmanLiftingFn]
    if config.lifting_functions.lifting_functions:
        lifting_functions = []
        for key, lf in config.lifting_functions.lifting_functions:
            lifting_functions.append((key, hydra.utils.instantiate(lf)))
    else:
        lifting_functions = None

    # Instantiate regressor
    regressor = hydra.utils.instantiate(config.regressor.regressor)
    if 't_step' in regressor.get_params().keys():
        regressor.set_params(t_step=dataset['t_step'])

    # Instantiate pipeline
    kp = pykoop.KoopmanPipeline(
        lifting_functions=lifting_functions,
        regressor=regressor,
    )

    # Log config
    logging.info(f'Config: {config}')

    # Figure out smallest number of training and validation timesteps
    n_steps_training, n_steps_validation = calc_n_steps(dataset)

    # Split training and validation data
    X_training, X_validation = split_training_validation(
        dataset, n_steps_training)

    # Perform cross-validation or a single-shot run
    if (('cv_config' in config.regressor.keys())
            and ('cv_params' in config.regressor.keys())):
        # Identify episode feature
        if dataset['episode_feature']:
            groups = X_training[:, 0]
        else:
            groups = np.zeros((X_training.shape[0], 1))
        # Split episodes into groups
        cv = sklearn.model_selection.GroupShuffleSplit(
            random_state=config.regressor.cv_config.seed,
            n_splits=config.regressor.cv_config.n_splits,
        )

        # Run BayesSearchCV or or GridSearchCV
        if config.regressor.cv_config.method == 'bayes':
            # BayesSearchCV
            cv_dict = {}
            for k, v in dict(config.regressor.cv_params).items():
                cv_dict[k] = tuple(v)
            gs = skopt.BayesSearchCV(
                kp,
                cv_dict,
                cv=cv,
                n_jobs=config.regressor.cv_config.n_jobs,
                n_iter=config.regressor.cv_config.n_iter,
                refit=True,
                verbose=4,
                # scoring=pykoop.KoopmanPipeline.make_scorer(
                #     n_steps=n_steps_validation),
                scoring=pykoop.KoopmanPipeline.make_scorer(n_steps=None),
                error_score=config.regressor.cv_config.error_score,
                optimizer_kwargs={
                    'base_estimator': 'GP',
                    'n_initial_points': 16,
                    'initial_point_generator': 'grid',
                })

            # Set up ``BayesSearchCV`` callbacks
            cb = [on_step_callback]
            if 'total_time' in config.regressor.cv_config.keys():
                cb.append(
                    skopt.callbacks.DeadlineStopper(
                        config.regressor.cv_config.total_time * 60 * 60))

            # Run cross-validation
            gs.fit(
                X_training,
                n_inputs=dataset['n_inputs'],
                episode_feature=dataset['episode_feature'],
                callback=cb,
                groups=groups,
            )
            fig, ax = plt.subplots()
            skopt.plots.plot_convergence(gs.optimizer_results_, ax=ax)
            fig.savefig(wd.joinpath(f'skopt_conv.png'))
        elif config.regressor.cv_config.method == 'halving':
            # Get ``n_states_out_``
            X_dummy = np.zeros_like(X_validation)
            kp_dummy = sklearn.base.clone(kp)
            kp_dummy.fit_transformers(
                X_dummy,
                n_inputs=dataset['n_inputs'],
                episode_feature=dataset['episode_feature'],
            )
            n_states_out = kp_dummy.n_states_out_
            # HalvingGridSearchCV
            params = get_gridsearchcv_params(config)
            # Configure cross-validation
            gs = sklearn.model_selection.HalvingGridSearchCV(
                kp,
                params,
                cv=cv,
                factor=config.regressor.cv_config.factor,
                resource='regressor__tsvd_shifted__truncation_param',
                max_resources=n_states_out,
                min_resources='exhaust',
                scoring=pykoop.KoopmanPipeline.make_scorer(n_steps=None),
                refit=True,
                # Create a new seed based on the original one
                random_state=(2 * config.regressor.cv_config.seed - 1),
                n_jobs=config.regressor.cv_config.n_jobs,
                verbose=4,
            )
            # Run cross-validation
            gs.fit(X_training,
                n_inputs=dataset['n_inputs'],
                episode_feature=dataset['episode_feature'],
                groups=groups,
            )
        else:
            # GridSearchCV
            params = get_gridsearchcv_params(config)
            # Configure cross-validation
            gs = sklearn.model_selection.GridSearchCV(
                kp,
                params,
                cv=cv,
                n_jobs=config.regressor.cv_config.n_jobs,
                refit=f'{n_steps_validation}_steps',
                verbose=4,
                scoring={
                    f'{n_steps_training}_steps':
                    pykoop.KoopmanPipeline.make_scorer(
                        n_steps=n_steps_training),
                    f'{n_steps_validation}_steps':
                    pykoop.KoopmanPipeline.make_scorer(
                        n_steps=n_steps_validation),
                    f'{n_steps_validation // 10}_steps':
                    pykoop.KoopmanPipeline.make_scorer(
                        n_steps=(n_steps_validation // 10)),
                },
            )
            # Run cross-validation
            gs.fit(X_training,
                n_inputs=dataset['n_inputs'],
                episode_feature=dataset['episode_feature'],
                groups=groups,
            )
        # Save results
        cv_results = pandas.DataFrame(gs.cv_results_)
        cv_results.to_csv(wd.joinpath('cv_results.csv'))
        estimator = gs.best_estimator_
    else:
        # Fit pipeline
        if not config.profile:
            kp.fit(X_training,
                   n_inputs=dataset['n_inputs'],
                   episode_feature=dataset['episode_feature'])
        else:
            # ``@profile`` decorator is defined by ``mprof``
            profile(kp.fit)(X_training,
                            n_inputs=dataset['n_inputs'],
                            episode_feature=dataset['episode_feature'])
        estimator = kp

    # Save best estimator
    with open(wd.joinpath('estimator.pickle'), 'wb') as f:
        pickle.dump(estimator, f)

    # Plot weights
    if (hasattr(estimator.regressor_, 'ss_ct_')
            and hasattr(estimator.regressor_, 'ss_dt_')):
        plot_weights(
            'weights',
            wd,
            estimator.regressor_.ss_ct_,
            estimator.regressor_.ss_dt_,
        )

    # Plot validation sets
    episodes = pykoop.split_episodes(
        X_validation, episode_feature=dataset['episode_feature'])
    for (i, X_i) in episodes:
        X_i_with_ep = np.hstack((
            i * np.ones((X_i.shape[0], 1)),
            X_i,
        ))
        plot_timeseries(f'timeseries_{i}', wd, X_i_with_ep, [estimator])
        plot_error(f'error_{i}', wd, X_i_with_ep, [estimator])
    plot_eigenvalues('eigenvalues', wd, [estimator])
    plot_matshow('matshow', wd, estimator)
    plot_mimo_bode('bode', wd, estimator, dataset['t_step'])
    plot_convergence('convergence', wd, estimator)
    # End timer
    end_time = time.monotonic()
    execution_time = end_time - start_time
    log_execution_time_and_notify(execution_time)


def on_step_callback(res):
    """Print iteration of ``BayesSearchCV``."""
    logging.info(f'Starting BayesSearchCV iteration: {res}')


def calc_n_steps(dataset: Dict) -> Tuple[int, int]:
    """Figure out smallest number of training and validation timesteps."""
    sizes_training = []
    sizes_validation = []
    episodes = pykoop.split_episodes(
        dataset['X'], episode_feature=dataset['episode_feature'])
    for (i, X_i) in episodes:
        if i in dataset['validation_episodes']:
            sizes_validation.append(X_i.shape[0])
        else:
            # If there's no episode feature, everything will fall here
            sizes_training.append(X_i.shape[0])
    n_steps_training = np.min(sizes_training)
    n_steps_validation = np.min(sizes_validation)
    if n_steps_validation > n_steps_training:
        logging.warning('More validation timesteps than training.')
    return (n_steps_training, n_steps_validation)


def split_training_validation(
        dataset: Dict, n_steps_training: int) -> Tuple[np.ndarray, np.ndarray]:
    """Split training and validation data."""
    if dataset['episode_feature']:
        training_idx = np.where(
            np.in1d(dataset['X'][:, 0], dataset['training_episodes']))[0]
        validation_idx = np.where(
            np.in1d(dataset['X'][:, 0], dataset['validation_episodes']))[0]
        X_training = dataset['X'][training_idx, :]
        X_validation = dataset['X'][validation_idx, :]
    else:
        X_training = dataset['X'][:(n_steps_training // 2), :]
        X_validation = dataset['X'][(n_steps_training // 2):, :]
    return (X_training, X_validation)


def get_gridsearchcv_params(
        config: omegaconf.DictConfig) -> Union[Dict, List[Dict]]:
    """Format parameters for ``GridSearchCV``."""
    params: Union[Dict, List[Dict]]
    if (type(config.regressor.cv_params) is omegaconf.listconfig.ListConfig):
        params = []
        for param in config.regressor.cv_params:
            params.append(dict(param))
    else:
        params = dict(config.regressor.cv_params)
    return params


def log_execution_time_and_notify(execution_time: float) -> None:
    """Log execution time and notify."""
    formatted_execution_time = datetime.timedelta(seconds=execution_time)
    logging.info(f'Execution time: {formatted_execution_time}')

    # Push notification if ``ntfy`` is installed and configured.
    cfg = hydra.core.hydra_config.HydraConfig.get().job.override_dirname
    status = f'Config: {cfg}\nExecution time: {formatted_execution_time}'
    try:
        subprocess.call(('ntfy', '--title', 'CV complete', 'send', status))
    except Exception:
        logging.warning('To enable push notifications, install `ntfy` from: '
                        'https://github.com/dschep/ntfy')


def plot_timeseries(path, wd, X_validation, estimators, labels=None):
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
        save_dict(path, wd, {
            'X_prediction': X_prediction,
            'X_validation': X_validation,
        })
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


def plot_error(path, wd, X_validation, estimators, labels=None):
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


def plot_weights(path, wd, ss_ct, ss_dt):
    """Plot Hinf weights."""
    fig, ax = plt.subplots(1, 2, constrained_layout=True)
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
    save_dict(
        path, wd, {
            'w_ct': w_ct,
            'H_ct': H_ct,
            'mag_ct': mag_ct,
            'w_dt': w_dt,
            'H_dt': H_dt,
            'mag_dt': mag_dt,
            'mag_dt_db': mag_dt_db,
        })
    # Save figure
    fig.savefig(wd.joinpath(f'{path}.png'))


def plot_eigenvalues(path, wd, estimators, labels=None):
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
        save_dict(path, wd, {
            'eigv': eigv,
            'eigv_mat': eigv_mag[idx],
        })

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
    for i in range(n_plt):
        ax[1, i] = fig.add_subplot(gs[1, i])
        ax[1, i].set_xlabel(r'$i$')
        ax[1, i].set_ylabel(r'$\|\lambda_i\|$')

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


def plot_matshow(path, wd, estimator, label=None):
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
    fig.colorbar(im)
    # Save figure
    save_dict(path, wd, {
        'U': U,
    })
    fig.savefig(wd.joinpath(f'{path}.png'))


def plot_mimo_bode(path, wd, estimator, t_step, label=None):
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
    save_dict(path, wd, {
        'f_samp': f_samp,
        'f_plot': f_plot,
        'mag': mag,
        'mag_db': mag_db,
    })
    fig.savefig(wd.joinpath(f'{path}.png'))


def plot_convergence(path, wd, estimator, label=None):
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
        ax.plot(obj)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Objective function value')
        # Save figure
        save_dict(path, wd, {
            'obj': obj,
        })
        fig.savefig(wd.joinpath(f'{path}.png'))


def save_dict(name, wd, dict_):
    """Save a dict."""
    path = wd.joinpath(f'{name}.pickle')
    with open(path, 'wb') as f:
        pickle.dump(dict_, f)


if __name__ == '__main__':
    main()
