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

## Reproducibility

This software was developed on Arch Linux with Python 3.10.1. The experiments
used in the corresponding paper were run on Windows 10 with Python 3.9.2.
However, `pykoop` supports any version of Python above 3.7.12.
All performance statistics are based on a PC with an Intel Core i7-10700K
processor using the [MOSEK solver](https://www.mosek.com/).

## Installation

To clone the repository and its
[submodule](https://github.com/ramvasudevan/soft-robot-koopman), which contains
the soft robot dataset, run
```sh
$ git clone --recurse-submodules git@github.com:decarsg/system_norm_koopman.git
```

You can install Python from your package manager or from the [official
website](https://www.python.org/downloads/). The recommended way to use Python
is through a [virtual
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
your terminal prompt in parentheses
```sh
(venv) $ 
```

To install the required dependencies, including `pykoop`, run
```sh
(venv) $ pip install -r ./requirements.txt
```

The LMI solver used, MOSEK, requires a license to use. You can request personal
academic license [here](https://www.mosek.com/products/academic-licenses/). You
will be emailed a license file which must be placed in `~/mosek/mosek.lic`[^2].

[^1]: On Windows, use `> \venv\Scripts\activate`.
[^2]: On Windows, place the license in `C:\Users\<USER>\mosek\mosek.lic`.

## Repository Layout

## Execution Time

| Experiment | Execution Time |
| ---------- | -------------- |
| `faster__polynomial2__edmd/` | |
| `faster__polynomial2__srconst_099/` | |
| `faster__polynomial2__srconst_1/` | |
| `soft_robot__polynomial3_delay1__edmd/` | |
| `soft_robot__polynomial3_delay1__hinf/` | |
| `soft_robot__polynomial3_delay1__hinf_dmdc/` | |
| `soft_robot__polynomial3_delay1__hinfw/` | |
| `soft_robot__polynomial3_delay1__srconst_0999/` | |
| `soft_robot__polynomial3_delay1__srconst_0999_dmdc/` | |
