"""
scenario.py
===========
Common configuration and helper routines shared by the four examples of the
simulation section. Keeping the scenario in a single module guarantees that
the network is trained and evaluated under identical conditions.

Scenario
--------
Carrier wavelength of 10 mm, corresponding to approximately 30 GHz in the
millimeter-wave band. The BS is a uniform linear array of 16 antennas with
half-wavelength spacing, placed at a height of 15 m. The RIS is a uniform
planar array of 20 x 20 elements with half-wavelength spacing, placed 60 m
away from the BS with its lower edge at the height of the array. The users
are located at ground level over a rectangular region in front of the panel.

The Rician factor used during training is LoS dominated, which matches the
propagation regime of the analysis. Each example may adopt a different
Rician factor at inference, which is the purpose of the robustness test of
Example 3.

Companion modules (repository root):
    beamris_base.py, beamris_sumrate.py, beamris_unfolding.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from beamris_sumrate import BeamRISSumRate
import beamris_unfolding as bu

# ===========================================================================
# 1. Scenario parameters
# ===========================================================================

WAVELENGTH = 10e-3          # carrier wavelength [m], about 30 GHz
L_ANT = 16                  # BS antennas
NX, NZ = 20, 20             # RIS elements per dimension
N_RIS = NX * NZ
BS_HEIGHT = 15.0            # height of the BS array [m]
RIS_DISTANCE = 60.0         # distance between the BS and the RIS panel [m]

NOISE_POWER = 2e-12         # noise power [W]
USER_POWER = 75e-3          # transmit power of each user [W]
M_USERS = 3                 # number of users, unless otherwise stated

# Region in which the users are placed, both in training and in evaluation
X_RANGE = (-20.0, 20.0)     # [m]
Y_RANGE = (45.0, 55.0)      # [m], the RIS panel being at y = RIS_DISTANCE

# Path-loss parameters, common to the three links
ALPHA_DB = -60.0            # reference attenuation at r0 [dB]
R0 = 1.0                    # reference distance [m]
ETA_UA, ETA_UR, ETA_RA = 3.0, 3.0, 2.0     # path-loss exponents

KAPPA_TRAIN = 1000.0        # Rician factor used in training [dB], LoS dominated

# Unfolded network and training
N_LAYERS = 40
N_EPOCHS = 500
BATCH_SIZE = 8
LEARNING_RATE = 3e-4

# Line-search parameters of the iterative benchmarks
LS_KWARGS = dict(initial_stepsize=1.0, max_iterations=1000, tolerance=1e-3,
                 ls_method="backtracking", ls_iterations=30,
                 ls_beta=0.5, ls_nu=1e-2, verbose=False)

# outputs are always written to <repository root>/outputs,
# independently of the directory from which a script is run
OUT_DIR = SCRIPT_DIR / "outputs"


# ===========================================================================
# 2. Geometry
# ===========================================================================

def build_geometry(kappa_db=KAPPA_TRAIN):
    """
    Return the dictionary of geometric and propagation parameters expected by
    the channel generator, for a given Rician factor.
    """
    antennas_pos = np.zeros((L_ANT, 3))
    antennas_pos[:, 0] = (WAVELENGTH / 2) * np.arange(L_ANT)
    antennas_pos[:, 2] = BS_HEIGHT

    xg = (WAVELENGTH / 2) * np.arange(NX)
    zg = (WAVELENGTH / 2) * np.arange(NZ) + BS_HEIGHT
    px, pz = np.meshgrid(xg, zg)
    reflects_pos = np.zeros((N_RIS, 3))
    reflects_pos[:, 0] = px.ravel()
    reflects_pos[:, 1] = RIS_DISTANCE
    reflects_pos[:, 2] = pz.ravel()

    return dict(wavel=WAVELENGTH, antennas_pos=antennas_pos,
                reflects_pos=reflects_pos,
                K_ua=kappa_db, alpha_ua=ALPHA_DB, eta_ua=ETA_UA,
                K_ur=kappa_db, alpha_ur=ALPHA_DB, eta_ur=ETA_UR,
                K_ra=kappa_db, alpha_ra=ALPHA_DB, eta_ra=ETA_RA, r0=R0)


def draw_user_positions(M):
    """Draw M user positions uniformly over the coverage region."""
    x = np.random.uniform(*X_RANGE, size=M)
    y = np.random.uniform(*Y_RANGE, size=M)
    return np.stack([x, y, np.zeros(M)], axis=1)


# ===========================================================================
# 3. Iterative solver with the selective cost function
# ===========================================================================

class BeamRISSelective(BeamRISSumRate):
    """
    Extension of BeamRISSumRate providing:
      (i)  the selective cost function and its analytic gradient, selected
           through algorithm='selective';
      (ii) the blockage set, stored in self.blocked_mask.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.blocked_mask = np.zeros(self.nusers, dtype=bool)

    # -- selective cost function and gradient ---------------------------
    def _ris_only_channels(self, phase):
        """RIS-only equivalent channels, columns h_m^r = C diag(exp(-j th)) b_m."""
        Phi = np.diag(np.exp(-1j * phase))
        return self.Hc_ra @ Phi @ self.Hc_ur

    def cost_function_selective(self, phase):
        """Selective cost function; reduces to the trace-based one if B is empty."""
        if not self.blocked_mask.any():
            return self.cost_function_modified(phase)
        Hr = self._ris_only_channels(phase)
        norms2 = np.real(np.sum(np.abs(Hr) ** 2, axis=0))
        tau = float(self.users_power @ norms2) + self.nantennas * self.noise_power
        cost = 0.0
        for m in np.flatnonzero(self.blocked_mask):
            cost += np.log2(1 - self.users_power[m] * norms2[m] / tau)
        return cost

    def gradient_selective(self, phase):
        """Analytic gradient of the selective cost function."""
        if not self.blocked_mask.any():
            return self.gradient_modified(phase)
        Q = np.diag(np.exp(1j * phase))
        B_ = self.Hc_ur
        C = self.Hc_ra
        Hr = C @ Q.conj().T @ B_
        D = Hr @ np.diag(self.users_power) @ B_.conj().T
        norms2 = np.real(np.sum(np.abs(Hr) ** 2, axis=0))
        tau = float(self.users_power @ norms2) + self.nantennas * self.noise_power

        diagQ = np.diag(Q)
        # the diagonal of C^H D is obtained in O(L N) operations, without
        # forming the N x N product
        diagCTD = np.sum(np.conj(C) * D, axis=0)
        total = np.zeros(self.nreflects, dtype=complex)
        for m in np.flatnonzero(self.blocked_mask):
            alpha_m = self.users_power[m] / (tau - self.users_power[m] * norms2[m])
            term1 = diagQ * (B_[:, m].conj() * (C.conj().T @ Hr[:, m]))
            term2 = (norms2[m] / tau) * diagQ * diagCTD
            total += alpha_m * np.imag(term1 - term2)
        return (2 / np.log(2)) * np.real(total)

    # -- optimization loop ----------------------------------------------
    def optimize(self, algorithm='standard', **kwargs):
        if algorithm != 'selective':
            return super().optimize(algorithm=algorithm, **kwargs)
        return self._loop(self.cost_function_selective,
                          self.gradient_selective, **kwargs)

    def _loop(self, cost_f, grad_func, initial_stepsize=1.0, max_iterations=1000,
              tolerance=0.0, direction_method='steepest_descent',
              ls_method='backtracking', ls_iterations=2000, ls_beta=0.5,
              ls_nu=1e-3, verbose=True):
        """Same structure as BeamRISSumRate.optimize, parameterized by the cost."""
        D = np.eye(self.nreflects)
        phase = np.zeros(self.nreflects)
        grad = grad_func(phase)
        scale = np.linalg.norm(grad)
        history = []
        k = 0
        while (np.linalg.norm(grad) / scale > tolerance) and (k < max_iterations):
            d = -D @ grad
            if ls_method == 'backtracking':
                mu = self._backtracking(cost_f, phase, grad, d, initial_stepsize,
                                        ls_iterations, ls_beta, ls_nu)
            else:
                mu = initial_stepsize
            history.append(-self.cost_function(phase))     # exact sum rate
            phase_new = phase + mu * d
            grad_new = grad_func(phase_new)
            if direction_method == 'steepest_descent':
                D = np.eye(self.nreflects)
            elif direction_method == 'quasi_newton':
                D = self._bfgs(mu, d, D, grad, grad_new)
            elif direction_method == 'damped_bfgs':
                D = self._damped_bfgs(mu, d, D, grad, grad_new)
            k += 1
            phase, grad = phase_new, grad_new

        R = self.autocorrelation_matrix(phase)
        Rinv = np.linalg.pinv(R)
        Phi = np.diag(np.exp(-1j * phase))
        W = np.zeros((self.nantennas, self.nusers), dtype=complex)
        for m in range(self.nusers):
            hm = self.Hc_ua[:, m] + self.Hc_ra @ Phi @ self.Hc_ur[:, m]
            W[:, m] = Rinv @ hm / (hm.conj() @ Rinv @ hm)
        return W, phase, np.array(history)


# ===========================================================================
# 4. Channel realizations
# ===========================================================================

def make_realization(geo, M, users_power, noise_power, n_blocked,
                     users_pos=None):
    """
    Build a BeamRISSelective object over the coverage region and impose
    blockage on n_blocked users by zeroing the corresponding columns of the
    direct-channel matrix.
    """
    if users_pos is None:
        users_pos = draw_user_positions(M)

    obj = BeamRISSelective(
        noise_power=noise_power, antennas_pos=geo["antennas_pos"],
        reflects_pos=geo["reflects_pos"], users_pos=users_pos,
        wavel=geo["wavel"], users_power=users_power,
        K_ua=geo["K_ua"], K_ur=geo["K_ur"], K_ra=geo["K_ra"],
        eta_ua=geo["eta_ua"], eta_ur=geo["eta_ur"], eta_ra=geo["eta_ra"],
        alpha_ua=geo["alpha_ua"], alpha_ur=geo["alpha_ur"],
        alpha_ra=geo["alpha_ra"], r0=geo["r0"])

    blocked = np.zeros(M, dtype=bool)
    if n_blocked > 0:
        idx = np.random.choice(M, size=n_blocked, replace=False)
        blocked[idx] = True
        obj.Hc_ua = np.array(obj.Hc_ua, copy=True)
        obj.Hc_ua[:, blocked] = 0.0
    obj.blocked_mask = blocked
    return obj


def to_torch(obj):
    """Channel matrices and powers of a realization, as PyTorch tensors."""
    c = lambda z: torch.tensor(z, dtype=torch.complex128)
    return (c(obj.Hc_ua), c(obj.Hc_ra), c(obj.Hc_ur),
            torch.tensor(obj.users_power, dtype=torch.float64))


def np_phase_to_torch(phase_np):
    """
    Convert a phase vector from the NumPy convention to the PyTorch one.

    The NumPy code base models the equivalent channel as
        h_m = a_m + C diag(exp(-j theta)) b_m,
    whereas the PyTorch implementation of the network uses the opposite sign.
    """
    return torch.tensor(-np.asarray(phase_np), dtype=torch.float64)


def assert_phase_convention(obj, noise_power, tol=1e-9):
    """Run-time check of the phase-convention conversion."""
    ph = np.random.uniform(-np.pi, np.pi, obj.nreflects)
    sr_np = -obj.cost_function(ph)
    A, C, B, pw = to_torch(obj)
    sr_t = bu.sum_rate_mvdr(A, C, B, np_phase_to_torch(ph), pw, noise_power).item()
    assert abs(sr_np - sr_t) < tol * max(1.0, abs(sr_np)), \
        f"phase convention mismatch: {sr_np} vs {sr_t}"


# ===========================================================================
# 5. Trained network
# ===========================================================================

def get_model(M=M_USERS, tag="main", seed=42, verbose=True):
    """
    Load the trained network from the cache, or train it offline and without
    supervision if the cache is not available. Training always uses the
    LoS-dominated Rician factor KAPPA_TRAIN.
    """
    OUT_DIR.mkdir(exist_ok=True)
    model = bu.UnfoldedRISSel(L=L_ANT, N=N_RIS, num_layers=N_LAYERS, init_mu=0.1)
    ckpt = OUT_DIR / f"model_{tag}.pt"
    cached = None
    if ckpt.exists():
        # a cached network is reused only when it matches the current
        # configuration, since the per-element gains are tied to the number
        # of reflecting elements and to the number of layers
        state = torch.load(ckpt)
        try:
            model.load_state_dict(state)
            cached = True
        except RuntimeError:
            cached = False
            if verbose:
                print(f"Cached network {ckpt.name} does not match the current "
                      f"configuration and will be retrained.")
    if cached:
        if verbose:
            print(f"Loading trained network: {ckpt}")
    else:
        if verbose:
            print(f"Training the network with M = {M} users "
                  f"(offline, unsupervised) ...")
        geo = build_geometry(KAPPA_TRAIN)
        bu.train_unfolded(model, geo, M, USER_POWER * np.ones(M), NOISE_POWER,
                          n_epochs=N_EPOCHS, batch_size=BATCH_SIZE,
                          lr=LEARNING_RATE, seed=seed, verbose=verbose)
        torch.save(model.state_dict(), ckpt)
        if verbose:
            print(f"Network saved to {ckpt}")
    model.eval()
    return model