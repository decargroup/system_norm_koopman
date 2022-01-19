# FASTER Dataset

This dataset was collected from the Fatigue Structural Testing Equipment
Research (FASTER) platform at the National Research Council of Canada (NRC).[^1]

In this dataset, an aluminum-composite beam is undergoing fatigue structural
testing. A controller computes the control input `u` from the force reference
`r` and the measured force `y`. The beam displacement `d` is also recorded. All
quantities are recorded at 128 Hz.

The columns of `faster.csv` are:

- `t`: time (s)
- `r`: force reference (force, unitless)
- `u`: control input (V)
- `y`: force output (force, unitless)
- `d`: displacement (mm)

See the cited paper for more details.[^1]

[^1]: R. Fortune, C. A. Beltempo, and J. R. Forbes, *System identification and
feedforward control of a fatigue structural testing rig: The single actuator
case*, IFAC-PapersOnLine, 52 (2019), pp. 382-387,
https://doi.org/10.1016/j.ifacol.2019.11.273
