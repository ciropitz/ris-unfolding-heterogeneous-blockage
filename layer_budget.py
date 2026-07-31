"""
layer_budget.py
===============
Design study supporting the number of layers adopted in the simulation setup.

The number of layers K fixes both the inference cost, of order K L N M, and
the number of trainable parameters, K (N + 1). This script measures the
resulting tradeoff and produces the figure and the numbers reported in the
setup subsection of the manuscript. It is not one of the four examples of the
simulation section.

All the methods are evaluated on a single set of channel realizations, built
once with the same generator used by Example 2, so that the results of this
study and of that example refer to comparable conditions. The exact cost
function with damped-BFGS directions is included as the reference against
which every fraction is reported.

Two measurements are produced.

  (a) RETRAINING SWEEP over K. One network is trained for every value of K
      under identical conditions and evaluated on the common set. This is the
      measurement that supports the choice of K, together with the execution
      time and the parameter count of each depth.

  (b) LAYER-WISE READOUT of one trained network. The sum rate attained by the
      intermediate configurations theta(k) is recorded along a single forward
      pass, which shows the shape of the trajectory inside the network. It is
      reported for illustration only: the step sizes of a network trained
      with K layers are tuned for a trajectory of that length, so the readout
      at layer k does not predict what a network trained with K = k attains,
      which is what part (a) measures.

Usage
-----
    python analysis/layer_budget.py

Training one network per value of K is expensive and is performed only once,
since every network is cached in outputs/. Deleting the cache forces
retraining.

Outputs
-------
    outputs/layer_budget.txt
    outputs/layer_budget_depth.png, .pdf        (part a)
    outputs/layer_budget_readout.png, .pdf      (part b)
    outputs/model_K{K}.pt                       (one network per value of K)
"""

import sys
import time
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

KAPPA = 1000.0              # LoS-dominated regime, as in Examples 1 and 2
K_LIST = [5, 10, 20, 30, 40, 60, 80, 100]
K_READOUT = 40              # depth used in the layer-wise readout of part (b)
N_EVAL = 50                 # channel realizations, shared by all the methods
EVAL_SEED = 123


# ===========================================================================
# 1. Common evaluation set
# ===========================================================================

def build_eval_set(geo, M, users_power, noise_power, n_eval=N_EVAL,
                   seed=EVAL_SEED):
    """
    Build the set of channel realizations shared by every method of this
    study. The number of blocked users is drawn uniformly over {0, ..., M},
    matching the distribution used in training.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    realizations = []
    for _ in range(n_eval):
        nb = int(np.random.randint(0, M + 1))
        obj = st.make_realization(geo, M, users_power, noise_power, nb)
        realizations.append((obj, st.to_torch(obj)))
    return realizations


# ===========================================================================
# 2. Evaluation of each method on the common set
# ===========================================================================

def evaluate_network(model, realizations, noise_power):
    """Average sum rate and execution time of a trained network."""
    rates, times = [], []
    for _, (A, C, B, pw) in realizations:
        t0 = time.perf_counter()
        with torch.no_grad():
            th = model(A, C, B, pw, noise_power)
        times.append(time.perf_counter() - t0)
        rates.append(bu.sum_rate_mvdr(A, C, B, th, pw, noise_power).item())
    return float(np.mean(rates)), 1e3 * float(np.mean(times))


def evaluate_anchor(realizations, noise_power):
    """
    Sum rate of the closed-form anchor alone, that is, of K = 0, which
    measures how much of the performance is already available before any
    layer is applied.
    """
    rates = []
    for _, (A, C, B, pw) in realizations:
        mask = bu.detect_blocked(A, C, B)
        with torch.no_grad():
            th = bu.compute_anchor(A, C, B, mask)
            rates.append(bu.sum_rate_mvdr(A, C, B, th, pw, noise_power).item())
    return float(np.mean(rates))


def evaluate_iterative(realizations, noise_power):
    """
    Average sum rate and execution time of the exact cost function with
    damped-BFGS directions, used as the reference of this study.
    """
    rates, times, iters = [], [], []
    for obj, (A, C, B, pw) in realizations:
        t0 = time.perf_counter()
        _, ph, hist = obj.optimize(algorithm="standard",
                                   direction_method="damped_bfgs",
                                   **st.LS_KWARGS)
        times.append(time.perf_counter() - t0)
        iters.append(len(hist))
        th = st.np_phase_to_torch(ph)
        rates.append(bu.sum_rate_mvdr(A, C, B, th, pw, noise_power).item())
    return (float(np.mean(rates)), 1e3 * float(np.mean(times)),
            float(np.mean(iters)))


# ===========================================================================
# 3. Networks
# ===========================================================================

def get_model_K(K, geo, M, users_power, noise_power, seed=42, verbose=True):
    """
    Load a network with K layers from the cache, or train it offline and
    without supervision. Mirrors scenario.get_model, with the layer count as
    an explicit argument and a dedicated cache file per depth.
    """
    st.OUT_DIR.mkdir(exist_ok=True)
    model = bu.UnfoldedRISSel(L=st.L_ANT, N=st.N_RIS, num_layers=K,
                              init_mu=0.1)
    ckpt = st.OUT_DIR / f"model_K{K}.pt"
    loaded = False
    if ckpt.exists():
        try:
            model.load_state_dict(torch.load(ckpt))
            loaded = True
            if verbose:
                print(f"  K = {K:3d}: loading {ckpt.name}")
        except RuntimeError:
            if verbose:
                print(f"  K = {K:3d}: cached network does not match the "
                      f"current configuration, retraining")
    if not loaded:
        if verbose:
            print(f"  K = {K:3d}: training ({st.N_EPOCHS} epochs) ...")
        bu.train_unfolded(model, geo, M, users_power, noise_power,
                          n_epochs=st.N_EPOCHS, batch_size=st.BATCH_SIZE,
                          lr=st.LEARNING_RATE, seed=seed, verbose=False)
        torch.save(model.state_dict(), ckpt)
    model.eval()
    return model


# ===========================================================================
# 4. Layer-wise readout
# ===========================================================================

def forward_trajectory(model, Hc_ua, Hc_ra, Hc_ur, pw, noise_power, mask=None):
    """
    Run the forward pass and return the phase configuration after every layer.

    Reproduces UnfoldedRISSel.forward_with_mask, storing theta(k) for
    k = 0, ..., K, where theta(0) is the closed-form anchor.
    """
    if mask is None:
        mask = bu.detect_blocked(Hc_ua, Hc_ra, Hc_ur)
    theta = bu.compute_anchor(Hc_ua, Hc_ra, Hc_ur, mask)
    traj = [theta.clone()]
    for layer in model.layers:
        grad = model._grad(Hc_ua, Hc_ra, Hc_ur, theta, pw, noise_power,
                           mask, create_graph=False)
        theta = layer(theta, grad)
        traj.append(theta.clone())
    return traj


def layer_readout(model, realizations, noise_power):
    """Average sum rate of the intermediate configurations theta(k)."""
    acc = np.zeros(model.num_layers + 1)
    for _, (A, C, B, pw) in realizations:
        with torch.no_grad():
            for k, th in enumerate(forward_trajectory(model, A, C, B, pw,
                                                      noise_power)):
                acc[k] += bu.sum_rate_mvdr(A, C, B, th, pw, noise_power).item()
    return acc / len(realizations)


# ===========================================================================
# 5. Figures
# ===========================================================================

def _save(fig, out_png):
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {out_png}")


def plot_depth(k_list, rates, times, anchor, reference, out_png):
    """(a) Sum rate and execution time as functions of the number of layers."""
    fig, ax = plt.subplots(figsize=(9 / 2.54, 6.5 / 2.54))
    ax.plot(k_list, rates, color="tab:blue", marker="o", markersize=4,
            linewidth=1.2, label="Proposed network")
    ax.axhline(reference, color="tab:red", linestyle="--", linewidth=1.2,
               label="Exact cost, damped BFGS")
    ax.axhline(anchor, color="0.6", linestyle=":", linewidth=1.2,
               label="Anchor only")
    ax.set_xlabel("Number of layers", fontsize=9, fontfamily="serif")
    ax.set_ylabel("Sum-rate capacity (bps/Hz)", fontsize=9,
                  fontfamily="serif")
    ax.set_xticks(k_list)
    ax.grid(True)
    ax.tick_params(labelsize=8)

    ax2 = ax.twinx()
    ax2.plot(k_list, times, color="tab:green", marker="^", markersize=4,
             linewidth=1.2, linestyle="-.", label="Execution time")
    ax2.set_ylabel("Execution time (ms)", fontsize=9, fontfamily="serif")
    ax2.tick_params(labelsize=8)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=6, loc="lower right",
              prop={"family": "serif", "size": 6})
    _save(fig, out_png)


def plot_readout(curve, out_png):
    """(b) Sum rate of the intermediate configurations along the layers."""
    fig, ax = plt.subplots(figsize=(9 / 2.54, 6.5 / 2.54))
    ax.plot(np.arange(len(curve)), curve, color="tab:blue", linewidth=1.2)
    ax.set_xlabel("Layer index", fontsize=9, fontfamily="serif")
    ax.set_ylabel("Sum-rate capacity (bps/Hz)", fontsize=9,
                  fontfamily="serif")
    ax.grid(True)
    ax.tick_params(labelsize=8)
    _save(fig, out_png)


# ===========================================================================
# 6. Study
# ===========================================================================

def main():
    torch.set_default_dtype(torch.float64)
    np.random.seed(42)
    torch.manual_seed(42)

    geo = st.build_geometry(KAPPA)
    M = st.M_USERS
    users_power = st.USER_POWER * np.ones(M)

    print(f"Building the evaluation set ({N_EVAL} channel realizations)")
    realizations = build_eval_set(geo, M, users_power, st.NOISE_POWER)
    st.assert_phase_convention(realizations[0][0], st.NOISE_POWER)

    print("Reference: exact cost function with damped-BFGS directions")
    ref_rate, ref_time, ref_iters = evaluate_iterative(realizations,
                                                       st.NOISE_POWER)
    print(f"    {ref_rate:.4f} bps/Hz, {ref_time:.2f} ms, "
          f"{ref_iters:.1f} iterations")

    anchor = evaluate_anchor(realizations, st.NOISE_POWER)
    print(f"Anchor only (K = 0): {anchor:.4f} bps/Hz")

    # ---- (a) retraining sweep over the number of layers ---------------
    print("\n(a) Retraining sweep over the number of layers")
    rates, times = [], []
    for K in K_LIST:
        model = get_model_K(K, geo, M, users_power, st.NOISE_POWER)
        sr, tm = evaluate_network(model, realizations, st.NOISE_POWER)
        rates.append(sr)
        times.append(tm)
        print(f"    K = {K:3d}: {sr:8.4f} bps/Hz ({100 * sr / ref_rate:5.1f}% "
              f"of the reference), {tm:7.2f} ms, "
              f"{K * (st.N_RIS + 1):6d} parameters")

    # linear fit of the execution time, confirming the O(K L N M) scaling
    slope, intercept = np.polyfit(K_LIST, times, 1)

    lines = ["=" * 76,
             "  LAYER BUDGET STUDY",
             f"  {N_EVAL} channel realizations, blockage cardinality drawn "
             f"uniformly over {{0, ..., {M}}}",
             "=" * 76, "",
             "  Reference, exact cost with damped BFGS",
             f"    sum rate        {ref_rate:8.4f} bps/Hz",
             f"    execution time  {ref_time:8.2f} ms",
             f"    iterations      {ref_iters:8.1f}", "",
             "(a) Retraining sweep over the number of layers",
             "    K".ljust(10) + "sum rate".rjust(12) + "of ref.".rjust(11)
             + "time (ms)".rjust(12) + "ms/layer".rjust(11)
             + "parameters".rjust(13),
             "    " + "-" * 65,
             "    0".ljust(10) + f"{anchor:12.4f}"
             + f"{100 * anchor / ref_rate:10.1f}%" + f"{0.0:12.2f}"
             + f"{0.0:11.3f}" + f"{0:13d}"]
    for K, sr, tm in zip(K_LIST, rates, times):
        lines.append(f"    {K:<6d}" + f"{sr:12.4f}"
                     + f"{100 * sr / ref_rate:10.1f}%" + f"{tm:12.2f}"
                     + f"{tm / K:11.3f}" + f"{K * (st.N_RIS + 1):13d}")
    lines += ["",
              f"    execution time fitted as {slope:.3f} ms per layer plus a "
              f"fixed overhead of {intercept:.3f} ms,",
              "    the latter accounting for the anchor and the energy rule"]

    # marginal gain per layer between consecutive depths
    lines += ["", "    marginal gain per layer (bps/Hz)"]
    prev_K, prev_r = 0, anchor
    for K, sr in zip(K_LIST, rates):
        lines.append(f"      {prev_K:3d} to {K:3d}: "
                     f"{(sr - prev_r) / (K - prev_K):8.4f}")
        prev_K, prev_r = K, sr

    # ---- (b) layer-wise readout ---------------------------------------
    print(f"\n(b) Layer-wise readout of the network with K = {K_READOUT}")
    model_ro = get_model_K(K_READOUT, geo, M, users_power, st.NOISE_POWER,
                           verbose=False)
    curve = layer_readout(model_ro, realizations, st.NOISE_POWER)
    final = curve[-1]
    lines += ["", f"(b) Layer-wise readout, K = {K_READOUT}",
              "    k".ljust(10) + "sum rate".rjust(12) + "of final".rjust(11)]
    step = max(1, K_READOUT // 10)
    for k in list(range(0, K_READOUT, step)) + [K_READOUT]:
        lines.append(f"    {k:<6d}" + f"{curve[k]:12.4f}"
                     + f"{100 * curve[k] / final:10.1f}%")
    idx = int(np.argmax(curve >= 0.99 * final))
    lines.append(f"    within 1% of the final value from layer {idx}")
    print(f"    within 1% of the final value from layer {idx}")

    (st.OUT_DIR / "layer_budget.txt").write_text("\n".join(lines))
    plot_depth(K_LIST, rates, times, anchor, ref_rate,
               st.OUT_DIR / "layer_budget_depth.png")
    plot_readout(curve, st.OUT_DIR / "layer_budget_readout.png")


if __name__ == "__main__":
    main()
