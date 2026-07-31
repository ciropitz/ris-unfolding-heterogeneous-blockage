"""
example2_blockage_sweep.py
==========================
Example 2: performance across blockage conditions.

Monte Carlo evaluation over the number of blocked users, from the fully
unobstructed case to the fully obstructed one. For every channel realization
the same channels are supplied to all methods, namely:

  * theta = 0, used as a reference level;
  * the iterative methods of the companion work [dsp2025] with the exact and
    the trace-based cost functions, both with damped-BFGS directions;
  * the iterative method with the proposed selective cost function;
  * the proposed unfolded network, evaluated in a single forward pass.

Two performance metrics are reported, the total sum rate and the average rate
of the blocked users, the latter being where the cost functions differ most.
The average number of iterations and the average execution time per channel
realization are also recorded, supporting the complexity discussion.

Modules (repository root):
    beamris_base.py, beamris_sumrate.py, beamris_unfolding.py,
    scenario.py

Outputs:
    outputs/example2_sum_rate.png, .pdf
    outputs/example2_blocked_rate.png, .pdf
    outputs/example2_blockage_sweep.txt
"""

import sys
from pathlib import Path

# the modules of the package live at the root of the repository
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scenario as st
import beamris_unfolding as bu

# Rician factor at inference, matching the regime adopted in training
KAPPA = 1000.0
N_SAMPLES = 50            # channel realizations per blockage cardinality

# Damped BFGS only, the strongest direction method of Example 1
METHODS = {
    "exact_bfgs":     ("standard",  "damped_bfgs"),
    "trace_bfgs":     ("modified",  "damped_bfgs"),
    "selective_bfgs": ("selective", "damped_bfgs"),
}
LABELS = {
    "zero":           r"$\theta = 0$",
    "exact_bfgs":     "Exact cost, damped BFGS",
    "trace_bfgs":     "Trace-based cost, damped BFGS",
    "selective_bfgs": "Selective cost, damped BFGS",
    "network":        "Proposed network",
}
STYLES = {
    "zero":           dict(color="0.6", marker="s", ls=":"),
    "exact_bfgs":     dict(color="tab:blue", marker="o", ls="-"),
    "trace_bfgs":     dict(color="tab:red", marker="^", ls="--"),
    "selective_bfgs": dict(color="tab:green", marker="D", ls="-."),
    "network":        dict(color="tab:orange", marker="v", ls="-"),
}


def run_sweep(model, geo, M, users_power, noise_power, n_samples, seed=123):
    """Sweep the blockage cardinality, evaluating all methods on the same channels."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    names = ["zero"] + list(METHODS.keys()) + ["network"]
    results = {n: {"sum": [[] for _ in range(M + 1)],
                   "blk": [[] for _ in range(M + 1)],
                   "time": [], "iters": []} for n in names}

    for nb in range(M + 1):
        print(f"  {nb} blocked users ({n_samples} realizations)")
        for t in range(n_samples):
            obj = st.make_realization(geo, M, users_power, noise_power, nb)
            if nb == 0 and t == 0:
                st.assert_phase_convention(obj, noise_power)
            A, C, B, pw = st.to_torch(obj)

            phases = {"zero": torch.zeros(obj.nreflects, dtype=torch.float64)}
            for name, (alg, direction) in METHODS.items():
                t0 = time.perf_counter()
                _, ph, hist = obj.optimize(algorithm=alg,
                                           direction_method=direction,
                                           **st.LS_KWARGS)
                results[name]["time"].append(time.perf_counter() - t0)
                results[name]["iters"].append(len(hist))
                phases[name] = st.np_phase_to_torch(ph)

            t0 = time.perf_counter()
            with torch.no_grad():
                phases["network"] = model(A, C, B, pw, noise_power)
            results["network"]["time"].append(time.perf_counter() - t0)
            results["network"]["iters"].append(model.num_layers)

            for name, ph in phases.items():
                rates = bu.per_user_rates_mvdr(A, C, B, ph, pw, noise_power)
                results[name]["sum"][nb].append(rates.sum())
                if nb > 0:
                    results[name]["blk"][nb].append(rates[obj.blocked_mask].mean())

    return results


def report(results, M, out_txt):
    """Print and save the numerical results."""
    nbs = np.arange(M + 1)
    lines = ["=" * 78,
             "  AVERAGE SUM RATE (bps/Hz), mean +- standard deviation",
             "=" * 78,
             "  method".ljust(24) + "".join(f"{nb} blocked".rjust(15) for nb in nbs),
             "-" * 78]
    for name in LABELS:
        row = f"  {name}".ljust(24)
        for nb in nbs:
            v = np.array(results[name]["sum"][nb])
            row += f"{v.mean():8.3f}+-{v.std():5.3f}"
        lines.append(row)

    lines += ["", "  AVERAGE RATE OF THE BLOCKED USERS (bps/Hz)", "-" * 78]
    for name in LABELS:
        row = f"  {name}".ljust(24) + "            ---"
        for nb in nbs[1:]:
            v = np.array(results[name]["blk"][nb])
            row += f"{v.mean():8.3f}+-{v.std():5.3f}"
        lines.append(row)

    lines += ["", "  COMPUTATIONAL COST PER CHANNEL REALIZATION", "-" * 78,
              "  method".ljust(24) + "iterations".rjust(14) + "time (ms)".rjust(14)]
    for name in LABELS:
        if not results[name]["time"]:
            continue
        it = np.mean(results[name]["iters"])
        tm = 1e3 * np.mean(results[name]["time"])
        lines.append(f"  {name}".ljust(24) + f"{it:14.1f}" + f"{tm:14.2f}")

    text = "\n".join(lines)
    print(text)
    out_txt.write_text(text)


def plot_sum_rate(results, M, out_png):
    """Average sum-rate capacity as a function of the number of blocked users."""
    nbs = np.arange(M + 1)
    fig, ax = plt.subplots(figsize=(9 / 2.54, 6.5 / 2.54))
    for name in LABELS:
        mu = [np.mean(results[name]["sum"][nb]) for nb in nbs]
        ax.plot(nbs, mu, label=LABELS[name], linewidth=1.2, markersize=4,
                **STYLES[name])
    ax.set_xlabel("Number of blocked users", fontsize=9, fontfamily="serif")
    ax.set_ylabel("Sum-rate capacity (bps/Hz)", fontsize=9, fontfamily="serif")
    ax.set_xticks(nbs)
    ax.grid(True)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=6, loc="best", prop={"family": "serif", "size": 6})
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {out_png}")


def plot_blocked_rate(results, M, out_png):
    """Average rate of the blocked users, for every blockage cardinality."""
    nbs = np.arange(M + 1)
    fig, ax = plt.subplots(figsize=(9 / 2.54, 6.5 / 2.54))
    for name in LABELS:
        mu = [np.mean(results[name]["blk"][nb]) for nb in nbs[1:]]
        ax.plot(nbs[1:], mu, label=LABELS[name], linewidth=1.2, markersize=4,
                **STYLES[name])
    ax.set_xlabel("Number of blocked users", fontsize=9, fontfamily="serif")
    # a short label is used here, since a longer one would exceed the height
    # of the axes; the caption states that the rate is averaged
    ax.set_ylabel("Blocked-user rate (bps/Hz)", fontsize=9, fontfamily="serif")
    ax.set_xticks(nbs[1:])
    ax.grid(True)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=6, loc="best", prop={"family": "serif", "size": 6})
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {out_png}")


def main():
    torch.set_default_dtype(torch.float64)
    np.random.seed(42)
    torch.manual_seed(42)

    model = st.get_model()
    geo = st.build_geometry(KAPPA)
    users_power = st.USER_POWER * np.ones(st.M_USERS)

    print("Monte Carlo sweep over the blockage cardinality:")
    results = run_sweep(model, geo, st.M_USERS, users_power, st.NOISE_POWER,
                        N_SAMPLES)
    report(results, st.M_USERS, st.OUT_DIR / "example2_blockage_sweep.txt")
    plot_sum_rate(results, st.M_USERS, st.OUT_DIR / "example2_sum_rate.png")
    plot_blocked_rate(results, st.M_USERS,
                      st.OUT_DIR / "example2_blocked_rate.png")


if __name__ == "__main__":
    main()