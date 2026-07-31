"""
example4_scheduling.py
======================
Example 4: per-slot scheduling in the RIS-dominated regime.

Under a rank-one RIS-to-BS channel the equivalent channels of the RIS-served
users are collinear, and fairness cannot be obtained in the spatial domain.
This script compares two ways of serving the blocked users:

  * simultaneous service with rate balancing, in which the RIS response is
    equalized among the served users. The per-user rate then saturates at
    log2(1 + 1/(|B| - 1)), independently of the aperture gain;

  * per-slot scheduling, in which the served set is rotated across scheduling
    slots in round-robin order and the trained network computes the phase
    configuration of every slot in a single forward pass, according to
    Algorithm 2 of the manuscript.

The long-term rates of both strategies are compared, together with the Jain
fairness index. The LoS-dominated Rician factor is adopted here, since the
rank-one condition underlying the analysis holds in that regime.

Modules (repository root):
    beamris_base.py, beamris_sumrate.py, beamris_unfolding.py,
    scenario.py

Outputs:
    outputs/example4_scheduling.png, .pdf
    outputs/example4_scheduling.txt
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

KAPPA = 1000.0            # LoS-dominated regime, in which C is rank one
N_SLOTS = 60              # scheduling window, multiple of the number of users
N_REALIZATIONS = 30


def balanced_phase(Hc_ua, Hc_ra, Hc_ur, users_power, noise_power, served,
                   n_iters=300, lr=5e-2):
    """
    Phase configuration that balances the RIS gains among the served users,
    which is the spatial-domain approach to fairness. The smallest gain is
    maximized while the dispersion among the gains is penalized.
    """
    N = Hc_ur.shape[0]
    theta = torch.zeros(N, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=lr)
    idx = torch.nonzero(served).flatten()
    for _ in range(n_iters):
        opt.zero_grad()
        Hr = bu.compute_ris_h(Hc_ra, Hc_ur, theta)
        gains = users_power[idx] * torch.sum(torch.abs(Hr[:, idx]) ** 2, dim=0)
        loss = -torch.log(gains.min() + 1e-30) + gains.std() / (gains.mean() + 1e-30)
        loss.backward()
        opt.step()
    return theta.detach()


def jain_index(rates):
    """Jain fairness index of a set of long-term rates."""
    r = np.asarray(rates, dtype=float)
    if np.allclose(r, 0):
        return 1.0
    return float(r.sum() ** 2 / (len(r) * np.sum(r ** 2)))


def run_realization(model, Hc_ua, Hc_ra, Hc_ur, users_power, noise_power,
                    blocked, n_slots):
    """Long-term rates of both strategies over one scheduling window."""
    M = Hc_ua.shape[1]
    blocked_idx = torch.nonzero(blocked).flatten().tolist()

    # per-slot scheduling with the round-robin policy
    acc = np.zeros(M)
    for t in range(n_slots):
        served = torch.zeros(M, dtype=torch.bool)
        served[blocked_idx[t % len(blocked_idx)]] = True
        with torch.no_grad():
            theta_t = model.forward_with_mask(Hc_ua, Hc_ra, Hc_ur, users_power,
                                              noise_power, served)
        acc += bu.per_user_rates_mvdr(Hc_ua, Hc_ra, Hc_ur, theta_t,
                                      users_power, noise_power)
    rates_rr = acc / n_slots

    # simultaneous service with rate balancing
    theta_bal = balanced_phase(Hc_ua, Hc_ra, Hc_ur, users_power, noise_power,
                               blocked)
    rates_bal = bu.per_user_rates_mvdr(Hc_ua, Hc_ra, Hc_ur, theta_bal,
                                       users_power, noise_power)
    return rates_rr, rates_bal


def main():
    torch.set_default_dtype(torch.float64)
    np.random.seed(42)
    torch.manual_seed(42)

    model = st.get_model()
    geo = st.build_geometry(KAPPA)
    users_power = st.USER_POWER * np.ones(st.M_USERS)
    M = st.M_USERS

    rr_all, bal_all = [], []
    print("Evaluating the scheduling window over the channel realizations:")
    for r in range(N_REALIZATIONS):
        A, C, B, pw, blocked = bu.generate_heterogeneous_sample(
            geo, M, users_power, st.NOISE_POWER, M)
        rr, bal = run_realization(model, A, C, B, pw, st.NOISE_POWER,
                                  blocked, N_SLOTS)
        rr_all.append(np.sort(rr)[::-1])
        bal_all.append(np.sort(bal)[::-1])
        if (r + 1) % 10 == 0:
            print(f"  {r + 1}/{N_REALIZATIONS}")

    rr_all = np.array(rr_all)
    bal_all = np.array(bal_all)
    rr_mean = rr_all.mean(axis=0)
    bal_mean = bal_all.mean(axis=0)
    r_max = np.log2(1 + 1 / (M - 1))

    lines = ["=" * 70,
             "  LONG-TERM PER-USER RATES (bps/Hz), users sorted by rate",
             "=" * 70,
             "  strategy".ljust(34) + "".join(f"user {i+1}".rjust(10)
                                              for i in range(M)),
             "-" * 70,
             "  Per-slot scheduling".ljust(34)
             + "".join(f"{v:10.3f}" for v in rr_mean),
             "  Simultaneous, rate balanced".ljust(34)
             + "".join(f"{v:10.3f}" for v in bal_mean),
             "",
             f"  Collinearity limit r_max             = {r_max:.3f} bps/Hz",
             f"  Jain index, per-slot scheduling      = "
             f"{np.mean([jain_index(x) for x in rr_all]):.4f}",
             f"  Jain index, simultaneous             = "
             f"{np.mean([jain_index(x) for x in bal_all]):.4f}",
             f"  Average long-term rate, scheduling   = {rr_mean.mean():.3f}",
             f"  Average long-term rate, simultaneous = {bal_mean.mean():.3f}"]
    text = "\n".join(lines)
    print("\n" + text)
    (st.OUT_DIR / "example4_scheduling.txt").write_text(text)

    fig, ax = plt.subplots(figsize=(9 / 2.54, 6.5 / 2.54))
    width = 0.35
    pos = np.arange(M)
    bars_rr = ax.bar(pos - width / 2, rr_mean, width, color="tab:green",
                     label="Per-slot scheduling")
    bars_bal = ax.bar(pos + width / 2, bal_mean, width, color="tab:red",
                      label="Simultaneous, rate balanced")
    line_rmax = ax.axhline(r_max, color="black", linestyle="--", linewidth=1.0,
                           label=r"$r_{\max}$")
    ax.set_xticks(pos)
    ax.set_xticklabels([f"User {i + 1}" for i in range(M)])
    ax.set_ylabel("Long-term rate (bps/Hz)", fontsize=9, fontfamily="serif")
    ax.tick_params(labelsize=8)
    ax.grid(True, axis="y")
    # headroom above the tallest bar, so that the legend does not overlap it
    ax.set_ylim(0, 1.45 * max(rr_mean.max(), bal_mean.max()))
    ax.legend(handles=[bars_rr, bars_bal, line_rmax], fontsize=7,
              loc="upper right", prop={"family": "serif", "size": 7})
    fig.tight_layout()
    fig.savefig(st.OUT_DIR / "example4_scheduling.png", dpi=300,
                bbox_inches="tight")
    fig.savefig(st.OUT_DIR / "example4_scheduling.pdf", bbox_inches="tight")
    print(f"\nFigure saved to {st.OUT_DIR / 'example4_scheduling.png'}")


if __name__ == "__main__":
    main()