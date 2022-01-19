# System Norm Regularization Methods for Koopman Operator Approximation

This repository contains the companion code for [System Norm Regularization
Methods for Koopman Operator Approximation](https://arxiv.org/abs/2110.09658).
All the code required to generate the paper's plots from raw data is included
here.

The regression methods detailed in the paper are implemented in
[`pykoop`](https://github.com/decarsg/pykoop), the authors' Koopman operator
identification library. This repository simply includes the appropriate calls
to `pykoop` to reproduce the paper's results.

This software relies on [`doit`](https://pydoit.org/) to automate plot
generation and [`hydra`](https://hydra.cc/) to automate experiment execution.

## Requirements

This software is compatible with Linux, macOS, and Windows. It was developed on
Arch Linux with Python 3.10.1, while the experiments used in the corresponding
paper were run on Windows 10 with Python 3.9.2. The `pykoop` library supports
any version of Python above 3.7.12. You can install Python from your package
manager or from the [official website](https://www.python.org/downloads/).

The performance statistics presented are from the [MOSEK
solver](https://www.mosek.com/) running with 16 threads on a PC with an Intel
Core i7-10700K processor and 64 GiB of RAM.

**Warning:** You probably cannot run this code on your laptop. The experiments
using the soft robot dataset are particularly demanding. A desktop with at
least 16 GiB of RAM is recommended.

## Installation

To clone the repository and its
[submodule](https://github.com/ramvasudevan/soft-robot-koopman), which contains
the soft robot dataset, run
```sh
$ git clone --recurse-submodules git@github.com:decarsg/system_norm_koopman.git
```

The recommended way to use Python is through a [virtual
environment](https://docs.python.org/3/library/venv.html). Create a virtual
environment (in this example, named `venv`) using
```sh
$ python -m virtualenv venv
```
Activate the virtual environment with[^1]
```sh
$ source ./venv/bin/activate
```
To use a specific version of Python in the virtual environment, instead use
```sh
$ source ./venv/bin/activate --python <PATH_TO_PYTHON_BINARY>
```
If the virtual environment is active, its name will appear at the beginning of
your terminal prompt in parentheses:
```sh
(venv) $ 
```

To install the required dependencies in the virtual environment, including
`pykoop`, run
```sh
(venv) $ pip install -r ./requirements.txt
```

The LMI solver used, MOSEK, requires a license to use. You can request personal
academic license [here](https://www.mosek.com/products/academic-licenses/). You
will be emailed a license file which must be placed in `~/mosek/mosek.lic`[^2].

[^1]: On Windows, use `> \venv\Scripts\activate`.
[^2]: On Windows, place the license in `C:\Users\<USER>\mosek\mosek.lic`.

## Usage

To automatically generate all the plots used in the paper, run
```sh
(venv) $ doit
```
in the repository root. This command will:

1. Preprocess the datasets in `datasets/` and place the outputs in
   `build/datasets/`
2. Run the necessary experiments using the datasets and place the outputs in
   `build/hydra_outputs/`
3. Profile the code and and place the results in `build/mprof_outputs/`
4. Generate the plots and place the results in `build/figures/`

This process can take upwards of **8 hours**. It requires at least **16 GiB of
RAM**. You can optionally add `-v2` to the command to print more detailed
information about the process as it runs

To execute just one task and its dependencies, run
```sh
(venv) $ doit <TASK_NAME>
```
To see a list of all available task names, run
```sh
(venv) $ doit list --all
```
For example, to generate only the FASTER eigenvalue plot, run
```sh
(venv) $ doit plot:faster_eig
```
The required experiments will be run automatically by `doit`.
The `experiment:*` and `profile:*` tasks used in the paper are:

| Task name | Execution time (hh:mm:ss) |
| --------- | ------------------------- |
| `experiment:faster__polynomial2__edmd` | 00:00:06 |
| `experiment:faster__polynomial2__srconst_099` | 00:00:39 |
| `experiment:faster__polynomial2__srconst_1` | 00:00:07 |
| `experiment:soft_robot__polynomial3_delay1__edmd` | 00:00:22 |
| `experiment:soft_robot__polynomial3_delay1__hinf` | 02:27:41 |
| `experiment:soft_robot__polynomial3_delay1__hinf_dmdc` | 00:23:41|
| `experiment:soft_robot__polynomial3_delay1__hinfw` | 04:00:20 |
| `experiment:soft_robot__polynomial3_delay1__srconst_0999` | 00:25:50|
| `experiment:soft_robot__polynomial3_delay1__srconst_0999_dmdc` | 00:05:27 |
| `profile:hinf` | 00:15:09 |
| `profile:hinf_dmdc` | 00:03:01 |
| `profile:srconst_0999` | 00:08:52 |
| `profile:srconst_0999_dmdc` | 00:02:03 |

Other experiments are listed by `doit list --all`, but they are not used.

If you have a pre-built copy of `build/hydra_outputs/` or other build products,
`doit` will think they are out-of-date and try to rebuild them. To prevent
this, run
```sh
(venv) $ doit reset-dep
```
after placing the folders in the right locations. This will force `doit` to
recognize the build products as up-to-date and prevent it from trying to
re-generate them. This is useful when moving the `build/` directory between
machines.

## Manually Calling Hydra

Hydra is responsible for running experiments from configuration files.
Normally, it is called by `doit`. If you want to run your own experiment with
this repository, you can manually call Hydra with (for example)
```sh
(venv) $ python ./run_experiment.py dataset=./build/datasets/faster.pickle \
> lifting_functions=polynomial2 regressor=hinf
```
Where `polynomial2` and `hinf` correspond to `yaml` files in
`config/lifting_functions/` and `config/regressor/` respectively. You can also
override `yaml` settings in the command line:
```sh
(venv) $ python ./run_experiment.py dataset=./build/datasets/faster.pickle \
> lifting_functions=polynomial2 regressor=hinf regressor.regressor.alpha=1
```
The Hydra outputs will appear in `outputs/<DATE>/<TIME>/`.

## Repository Layout

The files and folders of the repository are described here:

| Path | Description |
| --- | --- |
| `build/` | Contains all `doit` outputs, including plots. |
| `config/` | Contains configuration files for running experiments with Hydra. |
| `datasets/` | Contains raw datasets and their documentation. |
| `dodo.py` | Describes all of `doit`'s behaviour, like a `Makefile`. Also contains plotting code. |
| `run_experiment.py` | Script used by Hydra to run experiments from configuration files. |
| `requirements.txt` | Contains required Python packages with versions. |
| `LICENSE` | Repository license. |
| `README.md` | This file! |

If you want to know implementation details about the regressors presented in
the paper, look at the [`pykoop`](https://github.com/decarsg/pykoop/)
repository. If you're interested in the specific parameters these regressors
were called with, check out the `yaml` files in `config/`. Post-processing
calculations are done in `run_experiment.py`, while plotting code can be found
in `dodo.py`.
