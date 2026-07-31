"""
example1_convergence.py
=======================
Example 1: convergence behavior under heterogeneous blockage.

For a single channel realization drawn from the coverage region, with one
user having an obstructed direct link, this script compares:

  * the iterative gradient-based methods of the companion work [dsp2025],
    namely the exact and the trace-based cost functions, each with
    steepest-descent and damped-BFGS directions and backtracking line search;
  * the same iterative machinery applied to the proposed selective cost
    function, which separates the contribution of the cost function from
    that of the unfolding;
  * the proposed unfolded network, which produces its phase configuration in
    a single forward pass, shown as a horizontal line.

Run this script first, since it trains the network reused by the other
examples.

Modules (repository root):
    beamris_base.py, beamris_sumrate.py, beamris_unfolding.py,
    scenario.py

Outputs:
    outputs/example1_convergence.png, .pdf
    outputs/example1_convergence.txt
    outputs/model_main.pt   (trained network, reused by the other examples)
"""

import sys
from pathlib import Path

# the modules of the package live at the root of the repository
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scenario as st
import beamris_unfolding as bu

# Rician factor at inference for this example, matching the LoS-dominated
# regime adopted in training and in the analysis
KAPPA = 1000.0

# Iterative methods: (algorithm, direction, color, marker)
METHODS = {
    "std_sd":   ("standard",  "steepest_descent", (0.5, 0.5, 0.5), "o"),
    "std_bfgs": ("standard",  "damped_bfgs",      (0.5, 0.5, 0.5), None),
    "mod_sd":   ("modified",  "steepest_descent", (0.0, 0.0, 0.0), "o"),
    "mod_bfgs": ("modified",  "damped_bfgs",      (0.0, 0.0, 0.0), None),
    "sel_sd":   ("selective", "steepest_descent", (0.0, 0.5, 0.0), "o"),
    "sel_bfgs": ("selective", "damped_bfgs",      (0.0, 0.5, 0.0), None),
}
LABELS = {
    "std_sd":   "Exact cost, steepest descent",
    "std_bfgs": "Exact cost, damped BFGS",
    "mod_sd":   "Trace-based cost, steepest descent",
    "mod_bfgs": "Trace-based cost, damped BFGS",
    "sel_sd":   "Selective cost, steepest descent",
    "sel_bfgs": "Selective cost, damped BFGS",
}


def main():
    torch.set_default_dtype(torch.float64)
    np.random.seed(42)
    torch.manual_seed(42)

    model = st.get_model()
    geo = st.build_geometry(KAPPA)
    users_power = st.USER_POWER * np.ones(st.M_USERS)

    # Channel realization with one blocked user, drawn from the same region
    # used in training
    np.random.seed(7)
    obj = st.make_realization(geo, st.M_USERS, users_power, st.NOISE_POWER,
                              n_blocked=1)
    st.assert_phase_convention(obj, st.NOISE_POWER)
    print("User positions (m):\n", np.round(obj.users_pos, 2))
    print("Blocked users:", np.flatnonzero(obj.blocked_mask))

    curves = {}
    for name, (alg, direction, _, _) in METHODS.items():
        print(f"  running {name} ...")
        _, _, hist = obj.optimize(algorithm=alg, direction_method=direction,
                                  **st.LS_KWARGS)
        curves[name] = hist

    A, C, B, pw = st.to_torch(obj)
    with torch.no_grad():
        theta_net = model(A, C, B, pw, st.NOISE_POWER)
    sr_net = bu.sum_rate_mvdr(A, C, B, theta_net, pw, st.NOISE_POWER).item()

    # ------------------------------------------------------------------
    # Numerical summary
    # ------------------------------------------------------------------
    # the first entry of every history is the sum rate at theta = 0
    sr_init = curves["std_sd"][0]
    lines = ["Final sum-rate capacity (bps/Hz) and number of iterations", "-" * 60,
             f"  {'Initialization, theta = 0':38s} {sr_init:8.4f}       0"]
    for name, hist in curves.items():
        lines.append(f"  {LABELS[name]:38s} {hist[-1]:8.4f}   {len(hist):5d}")
    lines.append(f"  {'Proposed network (one forward pass)':38s} "
                 f"{sr_net:8.4f}   {st.N_LAYERS:5d}")
    text = "\n".join(lines)
    print("\n" + text)
    (st.OUT_DIR / "example1_convergence.txt").write_text(text)

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9 / 2.54, 6.5 / 2.54))
    for name, hist in curves.items():
        _, _, color, marker = METHODS[name]
        ax.plot(hist, color=color, linestyle="-", linewidth=1.2,
                marker=marker, markersize=3, markevery=40, label=LABELS[name])
    ax.axhline(sr_net, color=(0.8, 0.0, 0.0), linestyle="--", linewidth=1.2,
               label="Proposed network (one forward pass)")
    ax.set_xlabel("Iterations", fontsize=10, fontfamily="serif")
    ax.set_ylabel("Sum-rate capacity (bps/Hz)", fontsize=10, fontfamily="serif")
    # the legend position is chosen automatically among the standard anchors
    ax.legend(fontsize=6, loc="best", prop={"family": "serif", "size": 6})
    ax.tick_params(labelsize=8)
    ax.grid(True)
    for spine in ax.spines.values():
        spine.set_linewidth(1)
    fig.tight_layout()
    fig.savefig(st.OUT_DIR / "example1_convergence.png", dpi=300,
                bbox_inches="tight")
    fig.savefig(st.OUT_DIR / "example1_convergence.pdf", bbox_inches="tight")
    print(f"\nFigure saved to {st.OUT_DIR / 'example1_convergence.png'}")


if __name__ == "__main__":
    main()