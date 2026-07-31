import numpy as np


class BeamRISBase:
    """
    Base class for RIS-assisted beamforming.
    Handles channel modeling (Rician fading), SINR, and autocorrelation matrix.
    """

    def __init__(self, **kwargs):
        # Default parameters
        self.wavel = 1e-3
        self.K_ua = 1000.0
        self.alpha_ua = -60.0
        self.eta_ua = 2.0
        self.K_ur = 1000.0
        self.alpha_ur = -60.0
        self.eta_ur = 2.0
        self.K_ra = 1000.0
        self.alpha_ra = -60.0
        self.eta_ra = 2.0
        self.r0 = 1.0

        self.nantennas = 8
        self.antennas_pos = np.zeros((self.nantennas, 3))
        self.antennas_pos[:, 0] = (self.wavel / 2) * np.arange(self.nantennas)

        self.nreflects = 40
        self.reflects_pos = np.zeros((self.nreflects, 3))
        self.reflects_pos[:, 0] = (self.wavel / 2) * np.arange(self.nreflects)
        self.reflects_pos[:, 1] = 10.0

        self.users_pos = np.array([[0, 5, 0], [5, 5, 0], [5, 0, 0]], dtype=float)
        self.nusers = self.users_pos.shape[0]
        self.users_power = 100e-3 * np.ones(self.nusers)
        self.noise_power = 1e-12

        # Parse keyword arguments
        valid_args = {
            'antennas_pos', 'reflects_pos', 'users_pos', 'wavel', 'noise_power',
            'users_power', 'K_ua', 'alpha_ua', 'eta_ua', 'K_ur', 'alpha_ur',
            'eta_ur', 'K_ra', 'alpha_ra', 'eta_ra', 'r0'
        }
        for key, val in kwargs.items():
            if key not in valid_args:
                raise ValueError(f"Invalid argument name: {key}")
            setattr(self, key, val)
            if key == 'antennas_pos':
                self.nantennas = val.shape[0]
            elif key == 'reflects_pos':
                self.nreflects = val.shape[0]
            elif key == 'users_pos':
                self.nusers = val.shape[0]

        self._update_channels()

    def _update_channels(self):
        self.H_ua, self.Hc_ua = self._rician(
            self.users_pos, self.antennas_pos, self.K_ua, self.alpha_ua, self.eta_ua)
        self.H_ra, self.Hc_ra = self._rician(
            self.reflects_pos, self.antennas_pos, self.K_ra, self.alpha_ra, self.eta_ra)
        self.H_ur, self.Hc_ur = self._rician(
            self.users_pos, self.reflects_pos, self.K_ur, self.alpha_ur, self.eta_ur)

    def _rician(self, source, target, K_dB, alpha_dB, eta):
        """
        Generate Rician fading channel matrix.

        Parameters
        ----------
        source : (nsources, 3) array  - source positions
        target : (ntargets, 3) array  - target positions
        K_dB   : Rician factor [dB]
        alpha_dB : path-loss reference attenuation [dB]
        eta    : path-loss exponent

        Returns
        -------
        H  : (ntargets, nsources) steering matrix
        Hc : (ntargets, nsources) composite channel matrix
        """
        nsources = source.shape[0]
        ntargets = target.shape[0]

        # Steering matrix: H[t, s] = exp(j * k_s . p_t)
        norms = np.linalg.norm(source, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)          # avoid division by zero
        source_norm = source / norms                     # (nsources, 3)
        k = (2 * np.pi / self.wavel) * source_norm.T    # (3, nsources)
        H = np.exp(1j * (target @ k))                   # (ntargets, nsources)

        # Distance matrix
        r = np.zeros((ntargets, nsources))
        for s in range(nsources):
            r[:, s] = np.linalg.norm(source[s, :] - target, axis=1)

        K_lin    = 10 ** (K_dB / 10)
        alpha_lin = 10 ** (alpha_dB / 20)
        rho = alpha_lin * ((self.r0 / r) ** eta)        # (ntargets, nsources)

        los  = np.sqrt(K_lin * rho / (K_lin + 1)) * H
        nlos = np.sqrt(rho / (K_lin + 1)) * (
            (np.random.randn(ntargets, nsources) + 1j * np.random.randn(ntargets, nsources)) / np.sqrt(2)
        )
        Hc = los + nlos
        return H, Hc

    def sinr(self, w, phase, m):
        """
        Compute SINR for user m given beamforming vector w and RIS phase vector.

        Parameters
        ----------
        w     : (nantennas,) complex beamforming vector
        phase : (nreflects,) real phase vector [rad]
        m     : user index (0-based)
        """
        Phi = np.diag(np.exp(-1j * phase))
        hm = self.Hc_ua[:, m] + self.Hc_ra @ Phi @ self.Hc_ur[:, m]
        Rsoi = self.users_power[m] * np.outer(hm, hm.conj())

        Ripn = self.noise_power * np.eye(self.nantennas, dtype=complex)
        for i in range(self.nusers):
            if i != m:
                hi = self.Hc_ua[:, i] + self.Hc_ra @ Phi @ self.Hc_ur[:, i]
                Ripn += self.users_power[i] * np.outer(hi, hi.conj())

        sinr_val = np.real(w.conj() @ Rsoi @ w) / np.real(w.conj() @ Ripn @ w)
        return sinr_val

    def autocorrelation_matrix(self, phase):
        """
        Compute the received signal autocorrelation matrix R(phase).

        Parameters
        ----------
        phase : (nreflects,) real phase vector [rad]

        Returns
        -------
        R : (nantennas, nantennas) complex Hermitian matrix
        """
        Phi = np.diag(np.exp(-1j * phase))
        R = self.noise_power * np.eye(self.nantennas, dtype=complex)
        for i in range(self.nusers):
            h = self.Hc_ua[:, i] + self.Hc_ra @ Phi @ self.Hc_ur[:, i]
            R += self.users_power[i] * np.outer(h, h.conj())
        return R
