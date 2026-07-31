"""
beamris_unfolding.py
====================
Unfolded neural network for RIS-assisted beamforming in the MU-MISO uplink.

This module implements the proposed approach of the manuscript:

  1. SELECTIVE COST FUNCTION, indexed by the set B of RIS-served users,

         J_s = sum_{m in B} log2(1 - P_m ||h_m^r||^2 / tau)
         h_m^r = C Theta^H b_m                  (RIS-only channel)
         tau   = sum_i P_i ||h_i^r||^2 + L sigma^2

     which extends the trace-based reformulation to heterogeneous
     blockage. Boundary cases: an empty B reduces the expression to the
     trace-based cost evaluated with the full equivalent channels, and
     B = {1, ..., M} with all direct links obstructed makes it coincide
     with the trace-based cost, since h_m = h_m^r in that case.

  2. UNFOLDED NETWORK without an auxiliary input encoder. The
     initialization is the closed-form phase anchor and the trainable
     parameters reduce to the per-layer step size and per-element gains,
     in a total of num_layers * (N + 1) scalars.

  3. OFFLINE UNSUPERVISED TRAINING with |B| drawn uniformly over
     {0, ..., M} for every sample, using the negative MVDR sum rate as
     the loss function, so that no labeled solutions are required.

The set B is determined inside the forward pass from the observable
energies of the direct channels, unless it is specified externally by a
scheduler through forward_with_mask().

Phase convention: this module models the equivalent channel as
    h_m = a_m + C diag(exp(+j theta)) b_m,
which is the opposite sign of the NumPy implementation in
beamris_sumrate.py. Phase vectors exchanged between the two must be
negated, as done by scenario.np_phase_to_torch().

Dependencies: numpy, torch, beamris_base.py (repository root)
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from beamris_base import BeamRISBase


# ===========================================================================
# 1. SYSTEM FUNCTIONS
# ===========================================================================

def compute_all_h(Hc_ua, Hc_ra, Hc_ur, phase):
    """
    Full equivalent channels of all users,
        h_m = a_m + C diag(exp(j theta)) b_m.

    Notation of BeamRISBase:
        Hc_ua[:, m] -> a_m (L,),  Hc_ra -> C (L, N),  Hc_ur[:, m] -> b_m (N,)
    """
    phi_h = torch.exp(1j * phase.to(torch.float64))
    return Hc_ua + Hc_ra @ (phi_h.unsqueeze(1) * Hc_ur)


def compute_ris_h(Hc_ra, Hc_ur, phase):
    """RIS-only equivalent channels, h_m^r = C diag(exp(j theta)) b_m."""
    phi_h = torch.exp(1j * phase.to(torch.float64))
    return Hc_ra @ (phi_h.unsqueeze(1) * Hc_ur)


def compute_R(H_eff, users_power, noise_power):
    """Autocorrelation matrix R = sum_m P_m h_m h_m^H + sigma^2 I."""
    L = H_eff.shape[0]
    R = noise_power * torch.eye(L, dtype=torch.complex128)
    for m in range(H_eff.shape[1]):
        hm = H_eff[:, m]
        R = R + users_power[m] * torch.outer(hm, hm.conj())
    return R


def detect_blocked(Hc_ua, Hc_ra, Hc_ur):
    """
    Determine the blockage set from observable received energies,

        m in B  <=>  ||a_m||^2 < ||h_m^r||^2 at theta = 0,

    that is, whenever the direct link is weaker than the uncontrolled
    double-hop path through the RIS. Under obstruction the two energies
    differ by orders of magnitude, which makes the decision reliable.
    """
    N = Hc_ur.shape[0]
    Hr0 = compute_ris_h(Hc_ra, Hc_ur, torch.zeros(N, dtype=torch.float64))
    a_energy = torch.sum(torch.abs(Hc_ua) ** 2, dim=0)
    r_energy = torch.sum(torch.abs(Hr0) ** 2, dim=0)
    return a_energy < r_energy


def cost_J_selective(Hc_ua, Hc_ra, Hc_ur, phase, users_power, noise_power,
                     blocked_mask):
    """
    Selective cost function J_s(theta),

        J_s = sum_{m in B} log2(1 - P_m ||h_m^r||^2 / tau),
        tau = sum_i P_i ||h_i^r||^2 + L sigma^2.

    An empty set B reduces the expression to the trace-based cost
    evaluated with the full equivalent channels. Neither branch requires
    a matrix inversion.
    """
    L = Hc_ua.shape[0]

    if not bool(blocked_mask.any()):
        H_eff = compute_all_h(Hc_ua, Hc_ra, Hc_ur, phase)
        h_norms_sq = torch.sum(torch.abs(H_eff) ** 2, dim=0)
        trR = torch.dot(users_power, h_norms_sq) + L * noise_power
        J = torch.zeros(1, dtype=torch.float64)
        for m in range(Hc_ua.shape[1]):
            arg = torch.clamp(1.0 - users_power[m] * h_norms_sq[m] / trR,
                              min=1e-10)
            J = J + torch.log2(arg)
        return J.squeeze()

    Hr = compute_ris_h(Hc_ra, Hc_ur, phase)
    r_norms_sq = torch.sum(torch.abs(Hr) ** 2, dim=0)
    tau = torch.dot(users_power, r_norms_sq) + L * noise_power
    J = torch.zeros(1, dtype=torch.float64)
    for m in torch.nonzero(blocked_mask).flatten().tolist():
        arg = torch.clamp(1.0 - users_power[m] * r_norms_sq[m] / tau,
                          min=1e-10)
        J = J + torch.log2(arg)
    return J.squeeze()


def cost_J_exact(Hc_ua, Hc_ra, Hc_ur, phase, users_power, noise_power):
    """
    Exact cost function,
        J = sum_m log2(1 - P_m h_m^H R^{-1} h_m),
    provided for reference. It requires the inversion of R and is not
    used by the unfolded network.
    """
    H_eff = compute_all_h(Hc_ua, Hc_ra, Hc_ur, phase)
    R_inv = torch.linalg.inv(compute_R(H_eff, users_power, noise_power))
    J = torch.zeros(1, dtype=torch.float64)
    for m in range(H_eff.shape[1]):
        hm = H_eff[:, m]
        q = torch.real(hm.conj() @ R_inv @ hm)
        J = J + torch.log2(torch.clamp(1.0 - users_power[m] * q, min=1e-12))
    return J.squeeze()


def per_user_rates_mvdr(Hc_ua, Hc_ra, Hc_ur, phase, users_power,
                        noise_power, as_tensor=False):
    """Per-user rates log2(1 + SINR_m) attained with the MVDR beamformers."""
    H_eff = compute_all_h(Hc_ua, Hc_ra, Hc_ur, phase)
    R_inv = torch.linalg.inv(compute_R(H_eff, users_power, noise_power))
    rates = []
    for m in range(H_eff.shape[1]):
        hm = H_eff[:, m]
        q = torch.real(hm.conj() @ R_inv @ hm)
        q = torch.clamp(q, max=1.0 / users_power[m].item() - 1e-10)
        sinr = users_power[m] * q / (1.0 - users_power[m] * q + 1e-30)
        rates.append(torch.log2(1.0 + sinr))
    rates = torch.stack(rates)
    return rates if as_tensor else rates.detach().numpy()


def sum_rate_mvdr(Hc_ua, Hc_ra, Hc_ur, phase, users_power, noise_power):
    """Sum-rate capacity attained with the MVDR beamformers."""
    return per_user_rates_mvdr(Hc_ua, Hc_ra, Hc_ur, phase, users_power,
                               noise_power, as_tensor=True).sum()


def compute_anchor(Hc_ua, Hc_ra, Hc_ur, blocked_mask):
    """
    Closed-form phase anchor used as the initialization of the network,

        B nonempty: theta = -angle[C^H sum_{m in B} h_m^r] at theta = 0,
        B empty   : theta = -angle[C^H sum_m h_m] at theta = 0,

    which aligns the RIS response with the aggregate channel of the
    served users. It contains no trainable parameters.
    """
    N = Hc_ur.shape[0]
    phase0 = torch.zeros(N, dtype=torch.float64)
    if bool(blocked_mask.any()):
        Hr0 = compute_ris_h(Hc_ra, Hc_ur, phase0)
        h_sum = Hr0[:, blocked_mask].sum(dim=1)
    else:
        H0 = compute_all_h(Hc_ua, Hc_ra, Hc_ur, phase0)
        h_sum = H0.sum(dim=1)
    return -torch.angle(Hc_ra.conj().T @ h_sum)


# ===========================================================================
# 2. UNFOLDED LAYER
# ===========================================================================

class UnfoldedLayer(nn.Module):
    """
    One unfolded gradient iteration with trainable parameters,

        theta(k+1) = theta(k) - mu(k) diag[d(k)] grad J_s / ||grad J_s||_RMS,

    where mu(k) > 0 is a scalar step size and d(k) > 0 is a vector of
    per-element gains, both enforced positive through a softplus
    parameterization. The root-mean-square normalization of the gradient
    keeps the step magnitude well scaled along the whole trajectory.
    """

    def __init__(self, N, init_mu=0.1):
        super().__init__()
        init_val = float(np.log(np.expm1(max(init_mu, 1e-6))))
        self.log_mu = nn.Parameter(torch.tensor(init_val, dtype=torch.float64))
        self.log_d = nn.Parameter(torch.zeros(N, dtype=torch.float64))

    def forward(self, theta, grad_J):
        mu = nn.functional.softplus(self.log_mu)
        d = nn.functional.softplus(self.log_d)
        grad_scale = torch.sqrt(torch.mean(grad_J ** 2) + 1e-12)
        return theta - mu * d * (grad_J / grad_scale)


# ===========================================================================
# 3. UNFOLDED NETWORK
# ===========================================================================

class UnfoldedRISSel(nn.Module):
    """
    Unfolded network over the selective cost function, without encoder,

        theta(0)   = closed-form phase anchor,
        theta(k+1) = layer k applied to the gradient of J_s.

    The served set is determined inside the forward pass from the
    observable direct-channel energies, so that the network operates
    without oracle information, or is specified externally through
    forward_with_mask(). Trainable parameters: num_layers * (N + 1).
    """

    def __init__(self, L, N, num_layers=40, init_mu=0.1):
        super().__init__()
        self.L_ant = L
        self.N = N
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [UnfoldedLayer(N, init_mu=init_mu) for _ in range(num_layers)]
        )

    def _grad(self, Hc_ua, Hc_ra, Hc_ur, phase, pw, noise_power, mask,
              create_graph):
        """Gradient of the selective cost function with respect to the phases."""
        with torch.enable_grad():
            ph = phase.requires_grad_(True)
            J = cost_J_selective(Hc_ua, Hc_ra, Hc_ur, ph, pw, noise_power,
                                 blocked_mask=mask)
            grad = torch.autograd.grad(
                J, ph, retain_graph=create_graph, create_graph=create_graph
            )[0]
        return grad if create_graph else grad.detach()

    def forward_with_mask(self, Hc_ua, Hc_ra, Hc_ur, pw, noise_power, mask):
        """
        Forward pass with an externally specified served set.

        Used by the per-slot scheduler, which designates the set of users
        to be served in each slot instead of relying on the energy rule.
        """
        theta = compute_anchor(Hc_ua, Hc_ra, Hc_ur, mask)
        for layer in self.layers:
            grad = self._grad(Hc_ua, Hc_ra, Hc_ur, theta, pw, noise_power,
                              mask, create_graph=self.training)
            theta = layer(theta, grad)
        return theta

    def forward(self, Hc_ua, Hc_ra, Hc_ur, pw, noise_power):
        """Forward pass with the served set obtained from the energy rule."""
        mask = detect_blocked(Hc_ua, Hc_ra, Hc_ur)
        return self.forward_with_mask(Hc_ua, Hc_ra, Hc_ur, pw, noise_power, mask)


# ===========================================================================
# 4. CHANNEL REALIZATIONS WITH HETEROGENEOUS BLOCKAGE
# ===========================================================================

def generate_heterogeneous_sample(geo, M, users_power, noise_power,
                                  n_blocked, x_range=(-20, 20),
                                  y_range=(45, 55)):
    """
    Generate one channel realization with exactly n_blocked blocked users.

    The user positions are drawn uniformly over x_range by y_range at
    ground level. The channels are generated with all the direct links
    active, and the columns a_m of the randomly drawn blocked users are
    then set to zero.
    """
    x = np.random.uniform(*x_range, size=M)
    y = np.random.uniform(*y_range, size=M)
    users_pos = np.stack([x, y, np.zeros(M)], axis=1)

    ris = BeamRISBase(
        wavel=geo["wavel"],
        antennas_pos=geo["antennas_pos"],
        reflects_pos=geo["reflects_pos"],
        users_pos=users_pos,
        users_power=users_power,
        noise_power=noise_power,
        K_ua=geo["K_ua"], alpha_ua=geo["alpha_ua"], eta_ua=geo["eta_ua"],
        K_ur=geo["K_ur"], alpha_ur=geo["alpha_ur"], eta_ur=geo["eta_ur"],
        K_ra=geo["K_ra"], alpha_ra=geo["alpha_ra"], eta_ra=geo["eta_ra"],
        r0=geo.get("r0", 1),
    )

    Hc_ua = np.array(ris.Hc_ua, copy=True)
    blocked = np.zeros(M, dtype=bool)
    if n_blocked > 0:
        idx = np.random.choice(M, size=n_blocked, replace=False)
        blocked[idx] = True
        Hc_ua[:, blocked] = 0.0

    c = lambda z: torch.tensor(z, dtype=torch.complex128)
    return (c(Hc_ua), c(ris.Hc_ra), c(ris.Hc_ur),
            torch.tensor(ris.users_power, dtype=torch.float64),
            torch.tensor(blocked))


# ===========================================================================
# 5. OFFLINE UNSUPERVISED TRAINING
# ===========================================================================

def train_unfolded(model, geo, M, users_power, noise_power,
                   n_epochs=500, batch_size=8, lr=3e-4,
                   verbose=True, eval_interval=10, seed=None):
    """
    Train the unfolded network offline and without supervision.

    The loss is the negative sum-rate capacity attained with the MVDR
    beamformers, and every training sample is an independent channel
    realization with random user positions and with the number of blocked
    users drawn uniformly over {0, ..., M}. No labeled solutions and no
    runs of iterative solvers are required.
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr * 0.01
    )
    history = {"loss": [], "epoch": []}

    for epoch in range(1, n_epochs + 1):
        model.train()
        batch_loss = torch.zeros(1, dtype=torch.float64)

        for _ in range(batch_size):
            nb = int(np.random.randint(0, M + 1))
            Hc_ua, Hc_ra, Hc_ur, pw, _ = generate_heterogeneous_sample(
                geo, M, users_power, noise_power, nb
            )
            theta_K = model(Hc_ua, Hc_ra, Hc_ur, pw, noise_power)
            sr = sum_rate_mvdr(Hc_ua, Hc_ra, Hc_ur, theta_K, pw, noise_power)
            batch_loss = batch_loss - sr

        loss = batch_loss / batch_size
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()

        history["loss"].append(loss.item())
        history["epoch"].append(epoch)
        if verbose and (epoch % eval_interval == 0 or epoch == 1):
            print(f"  epoch {epoch:4d}/{n_epochs}  |  loss = {loss.item():.6f}")

    return history
