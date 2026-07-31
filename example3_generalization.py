"""
example3_generalization.py
==========================
Example 3: dependence of the trained network on the system parameters.

The trainable parameters of the unfolded network are the per-layer step sizes
and the per-element gains, the latter having the dimension of the RIS panel.
Neither the number of users nor the transmit powers, the noise power or the
Rician factor appears in the dimension of any trainable parameter, which
raises the question of how far a single trained network can be reused. Four
tests quantify this dependence.

  (a) Joint scaling of the transmit powers and of the noise power. The
      gradient of the selective cost function is homogeneous of degree zero
      under such a scaling, and the phase configuration is therefore
      invariant. The test verifies the invariance numerically.

  (b) Scaling of the transmit powers alone, which changes the operating
      signal-to-noise ratio. The sensitivity is measured as a function of the
      interference-to-noise ratio.

  (c) Mismatch in the number of users between training and inference, using
      two networks trained with different numbers of users.

  (d) Mismatch in the Rician factor between training and inference, with the
      network trained in the LoS-dominated regime and evaluated over a range
      of Rician factors.

Modules (repository root):
    beamris_base.py, beamris_sumrate.py, beamris_unfolding.py,
    scenario.py

Outputs:
    outputs/example3_power.png, .pdf     (test b)
    outputs/example3_users.png, .pdf     (test c)
    outputs/example3_rician.png, .pdf    (test d)
    outputs/example3_generalization.txt
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

M_TRAIN_A, M_TRAIN_B = 3, 5              # numbers of users used in training
M_EVAL = [2, 3, 4, 5, 6]                 # numbers of users used in evaluation
KAPPA_EVAL = [0.0, 5.0, 10.0, 20.0, 30.0, 1000.0]   # Rician factors [dB]
NOISE_LEVELS = [2e-9, 2e-10, 2e-11, 2e-12, 2e-13, 2e-14, 2e-15, 2e-16]
POWER_FACTOR = 100.0                     # scaling applied in tests (a) and (b)
N_SAMPLES = 30


# ===========================================================================
# 1. Tests
# ===========================================================================

def test_joint_scaling(model, geo, M, users_power, noise_power,
                       factors=(1e-3, 1e-1, 1e1, 1e3), n_samples=20, seed=1):
    """(a) Largest phase deviation under a joint scaling of powers and noise."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    worst = 0.0
    for _ in range(n_samples):
        nb = int(np.random.randint(0, M + 1))
        A, C, B, pw, _ = bu.generate_heterogeneous_sample(
            geo, M, users_power, noise_power, nb)
        with torch.no_grad():
            ref = model(A, C, B, pw, noise_power)
            for c in factors:
                dev = (model(A, C, B, pw * c, noise_power * c) - ref).abs().max()
                worst = max(worst, dev.item())
    return worst


def test_power_scaling(model, geo, M, users_power, noise_levels,
                       factor=POWER_FACTOR, n_samples=20, seed=2):
    """(b) Phase deviation under a power-only scaling, versus the INR."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    zero = torch.zeros(st.N_RIS, dtype=torch.float64)
    inr_list, dev_list = [], []
    for noise_power in noise_levels:
        inr_acc, dev_acc = [], []
        for _ in range(n_samples):
            A, C, B, pw, _ = bu.generate_heterogeneous_sample(
                geo, M, users_power, noise_power, M)
            Hr = bu.compute_ris_h(C, B, zero)
            energy = (pw * torch.sum(torch.abs(Hr) ** 2, dim=0)).sum().item()
            inr_acc.append(energy / (st.L_ANT * noise_power))
            with torch.no_grad():
                ref = model(A, C, B, pw, noise_power)
                dev = (model(A, C, B, pw * factor, noise_power) - ref).abs().max()
            dev_acc.append(dev.item())
        inr_list.append(np.mean(inr_acc))
        dev_list.append(np.mean(dev_acc))
    return np.array(inr_list), np.array(dev_list)


def evaluate_sum_rate(model, geo, M_eval, users_power, noise_power,
                      n_samples=N_SAMPLES, seed=99):
    """Average sum rate of a model and of the theta = 0 reference."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    zero = torch.zeros(st.N_RIS, dtype=torch.float64)
    net, base = [], []
    for _ in range(n_samples):
        nb = int(np.random.randint(0, M_eval + 1))
        A, C, B, pw, _ = bu.generate_heterogeneous_sample(
            geo, M_eval, users_power, noise_power, nb)
        with torch.no_grad():
            th = model(A, C, B, pw, noise_power)
        net.append(bu.per_user_rates_mvdr(A, C, B, th, pw, noise_power).sum())
        base.append(bu.per_user_rates_mvdr(A, C, B, zero, pw, noise_power).sum())
    return float(np.mean(net)), float(np.mean(base))


# ===========================================================================
# 2. Experiment
# ===========================================================================

def main():
    torch.set_default_dtype(torch.float64)
    np.random.seed(42)
    torch.manual_seed(42)

    geo_train = st.build_geometry(st.KAPPA_TRAIN)
    pw3 = st.USER_POWER * np.ones(M_TRAIN_A)
    lines = []

    net_a = st.get_model(M=M_TRAIN_A, tag="main", seed=42)
    net_b = st.get_model(M=M_TRAIN_B, tag=f"M{M_TRAIN_B}", seed=43)

    # ---- (a) joint scaling -------------------------------------------
    print("\n(a) Joint scaling of the transmit powers and the noise power")
    worst = test_joint_scaling(net_a, geo_train, M_TRAIN_A, pw3, st.NOISE_POWER)
    lines += ["(a) Joint scaling of powers and noise power",
              f"    largest phase deviation over four decades: {worst:.3e} rad"]
    print(f"    largest phase deviation: {worst:.3e} rad")

    # ---- (b) power-only scaling --------------------------------------
    print("\n(b) Scaling of the transmit powers alone")
    inr, dev = test_power_scaling(net_a, geo_train, M_TRAIN_A, pw3, NOISE_LEVELS)
    lines += ["", f"(b) Power-only scaling (factor {POWER_FACTOR:.0f})",
              "    interference-to-noise ratio   phase deviation (rad)"]
    for a, b in zip(inr, dev):
        lines.append(f"    {a:24.3e} {b:20.3e}")
        print(f"    INR = {a:.3e}   deviation = {b:.3e} rad")

    # ---- (c) mismatch in the number of users -------------------------
    print("\n(c) Mismatch in the number of users")
    rows, base_vals = {}, []
    for tag, model in [(f"trained M={M_TRAIN_A}", net_a),
                       (f"trained M={M_TRAIN_B}", net_b)]:
        vals = []
        for Me in M_EVAL:
            sr, base = evaluate_sum_rate(model, geo_train, Me,
                                         st.USER_POWER * np.ones(Me),
                                         st.NOISE_POWER)
            vals.append(sr)
            if tag.endswith(str(M_TRAIN_A)):
                base_vals.append(base)
        rows[tag] = vals
        print(f"    {tag}: " + " ".join(f"{v:.3f}" for v in vals))
    lines += ["", "(c) Sum rate (bps/Hz) versus the number of users at inference",
              "    model".ljust(22) + "".join(f"M={m}".rjust(10) for m in M_EVAL)]
    for tag in rows:
        lines.append(f"    {tag}".ljust(22)
                     + "".join(f"{v:10.3f}" for v in rows[tag]))
    lines.append("    theta = 0".ljust(22)
                 + "".join(f"{v:10.3f}" for v in base_vals))

    # ---- (d) mismatch in the Rician factor ---------------------------
    print("\n(d) Mismatch in the Rician factor")
    kappa_net, kappa_base = [], []
    for kap in KAPPA_EVAL:
        geo_k = st.build_geometry(kap)
        sr, base = evaluate_sum_rate(net_a, geo_k, M_TRAIN_A, pw3, st.NOISE_POWER)
        kappa_net.append(sr)
        kappa_base.append(base)
        print(f"    kappa = {kap:7.1f} dB: network {sr:.3f}, theta = 0 {base:.3f}")
    lines += ["", "(d) Sum rate (bps/Hz) versus the Rician factor at inference",
              "    kappa (dB)".ljust(22)
              + "".join(f"{k:10.0f}" for k in KAPPA_EVAL),
              "    network".ljust(22)
              + "".join(f"{v:10.3f}" for v in kappa_net),
              "    theta = 0".ljust(22)
              + "".join(f"{v:10.3f}" for v in kappa_base)]

    (st.OUT_DIR / "example3_generalization.txt").write_text("\n".join(lines))

    # ------------------------------------------------------------------
    # Figures, each written to a separate file
    # ------------------------------------------------------------------
    plot_power_sensitivity(inr, dev, st.OUT_DIR / "example3_power.png")
    plot_users(rows, base_vals, st.OUT_DIR / "example3_users.png")
    plot_rician(kappa_net, kappa_base, st.OUT_DIR / "example3_rician.png")


# ===========================================================================
# 3. Figures
# ===========================================================================

def _save(fig, out_png):
    """Save a figure in both raster and vector formats and close it."""
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {out_png}")


def plot_power_sensitivity(inr, dev, out_png):
    """(b) Phase deviation caused by a power-only scaling, versus the INR."""
    fig, ax = plt.subplots(figsize=(9 / 2.54, 6.5 / 2.54))
    ax.loglog(inr, np.maximum(dev, 1e-18), color="tab:blue", marker="o",
              markersize=4, linewidth=1.2)
    ax.set_xlabel("Interference-to-noise ratio", fontsize=9, fontfamily="serif")
    ax.set_ylabel("Phase deviation (rad)", fontsize=9, fontfamily="serif")
    ax.grid(True, which="both")
    ax.tick_params(labelsize=8)
    _save(fig, out_png)


def plot_users(rows, base_vals, out_png):
    """(c) Sum rate versus the number of users adopted at inference."""
    fig, ax = plt.subplots(figsize=(9 / 2.54, 6.5 / 2.54))
    # the two curves nearly coincide, hence the second one is drawn thinner,
    # dashed and with open markers, so that the first remains visible
    ax.plot(M_EVAL, rows[f"trained M={M_TRAIN_A}"], color="tab:blue",
            marker="o", markersize=6, linewidth=2.6,
            label=f"trained M={M_TRAIN_A}")
    ax.plot(M_EVAL, rows[f"trained M={M_TRAIN_B}"], color="tab:green",
            marker="D", markersize=3.5, linewidth=1.1, linestyle="--",
            markerfacecolor="none", markeredgewidth=1.1,
            label=f"trained M={M_TRAIN_B}")
    ax.plot(M_EVAL, base_vals, color="0.6", marker="s", linestyle=":",
            linewidth=1.2, markersize=4, label=r"$\theta = 0$")
    ax.set_xlabel("Number of users at inference", fontsize=9, fontfamily="serif")
    ax.set_ylabel("Sum-rate capacity (bps/Hz)", fontsize=9, fontfamily="serif")
    ax.set_xticks(M_EVAL)
    ax.grid(True)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7, loc="best", prop={"family": "serif", "size": 7})
    _save(fig, out_png)


def plot_rician(kappa_net, kappa_base, out_png):
    """(d) Sum rate versus the Rician factor adopted at inference."""
    fig, ax = plt.subplots(figsize=(9 / 2.54, 6.5 / 2.54))
    kx = np.arange(len(KAPPA_EVAL))
    ax.plot(kx, kappa_net, color="tab:blue", marker="o", markersize=4,
            linewidth=1.2, label="Proposed network")
    ax.plot(kx, kappa_base, color="0.6", marker="s", linestyle=":",
            markersize=4, linewidth=1.2, label=r"$\theta = 0$")
    ax.set_xticks(kx)
    ax.set_xticklabels([f"{k:.0f}" for k in KAPPA_EVAL])
    ax.set_xlabel("Rician factor at inference (dB)", fontsize=9,
                  fontfamily="serif")
    ax.set_ylabel("Sum-rate capacity (bps/Hz)", fontsize=9, fontfamily="serif")
    ax.grid(True)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7, loc="best", prop={"family": "serif", "size": 7})
    _save(fig, out_png)


if __name__ == "__main__":
    main()