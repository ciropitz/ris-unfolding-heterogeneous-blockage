# Deep unfolding for RIS beamforming under heterogeneous blockage

Simulation code accompanying the manuscript. The scripts reproduce every
figure, table, and number reported in the simulation section.

## Contents

```
.
├── beamris_base.py            channel model, SINR, autocorrelation matrix
├── beamris_sumrate.py         exact and trace-based cost functions, iterative solvers
├── beamris_unfolding.py       selective cost function, phase anchor, unfolded network
├── scenario.py                shared scenario configuration and helper routines
├── examples/
│   ├── example1_convergence.py       Fig. 3
│   ├── example2_blockage_sweep.py    Figs. 4 and 5 Table 1
│   ├── example3_generalization.py    Figs. 6, 7, and 8
│   └── example4_scheduling.py        Fig. 9
├── analysis/
│   └── layer_budget.py        design study for the number of layers - Fig. 2
├── outputs/                   generated figures, tables, and trained networks
├── requirements.txt
└── LICENSE
```

## Requirements

Python 3.13 with the packages listed in `requirements.txt`.

```
pip install -r requirements.txt
```

The results reported in the manuscript were obtained on an Apple M4 processor
with 16 GB of RAM, with NumPy linked against the Apple Accelerate (vecLib)
BLAS and LAPACK backend. All the scripts run on the central processing unit
and require no accelerator.

## Reproducing the results

Run the examples in order. The first one trains the network that the others
reuse, so it must be executed before the remaining scripts.

```
python examples/example1_convergence.py
python examples/example2_blockage_sweep.py
python examples/example3_generalization.py
python examples/example4_scheduling.py
```

The design study of the layer budget is independent of the examples and is
considerably more expensive, since it trains one network per depth.

```
python analysis/layer_budget.py
```

Every script writes its figures, in PNG and PDF, together with a text file
containing the corresponding numerical results, to `outputs/`. Trained
networks are cached in the same directory and are reused on subsequent runs.
Deleting a checkpoint forces retraining.

## Notes

Training is offline and unsupervised. The loss is the negative sum-rate
capacity attained with the MVDR beamformers, so no labeled solutions and no
runs of iterative solvers are required.

The NumPy and the PyTorch code bases adopt opposite sign conventions for the
RIS phases. Phase vectors exchanged between them must be negated, which is
handled by `scenario.np_phase_to_torch` and verified at run time by
`scenario.assert_phase_convention`.

## Citation

To be completed on acceptance.