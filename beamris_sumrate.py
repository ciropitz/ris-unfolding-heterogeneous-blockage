import numpy as np
from beamris_base import BeamRISBase


class BeamRISSumRate(BeamRISBase):
    """
    Sum-rate maximization for RIS-assisted MU-MISO uplink.
    Extends BeamRISBase with gradient-based phase optimization.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    # Objective / cost
    # ------------------------------------------------------------------

    def sum_rate(self, W, phase):
        """Compute sum-rate capacity [bps/Hz] given beamforming matrix W and phase."""
        val = 0.0
        for m in range(self.nusers):
            val += np.log2(1 + self.sinr(W[:, m], phase, m))
        return val

    def cost_function(self, phase):
        """
        Cost function (to be minimized, i.e. negative sum-rate surrogate).
        Uses exact inverse of R.
        """
        Rinv = np.linalg.pinv(self.autocorrelation_matrix(phase))
        Phi  = np.diag(np.exp(-1j * phase))
        cost = 0.0
        for m in range(self.nusers):
            hm = self.Hc_ua[:, m] + self.Hc_ra @ Phi @ self.Hc_ur[:, m]
            cost += np.log2(1 - np.real(self.users_power[m] * hm.conj() @ Rinv @ hm))
        return cost

    def cost_function_modified(self, phase):
        """Modified cost function using trace(R) instead of full matrix inverse."""
        R   = self.autocorrelation_matrix(phase)
        Phi = np.diag(np.exp(-1j * phase))
        trR = np.trace(R)
        cost = 0.0
        for m in range(self.nusers):
            hm = self.Hc_ua[:, m] + self.Hc_ra @ Phi @ self.Hc_ur[:, m]
            cost += np.log2(1 - np.real(self.users_power[m] * (hm.conj() @ hm) / trR))
        return cost

    # ------------------------------------------------------------------
    # Gradients
    # ------------------------------------------------------------------

    def gradient(self, phase):
        """Gradient of cost_function w.r.t. phase."""
        Rinv = np.linalg.pinv(self.autocorrelation_matrix(phase))
        Q    = np.diag(np.exp(1j * phase))
        B    = self.Hc_ur                           # (nreflects, nusers)
        C    = self.Hc_ra                           # (nantennas, nreflects)
        H    = self.Hc_ua + C @ Q.conj().T @ B     # (nantennas, nusers)  [Q^H = conj(Q) for diagonal unitary]
        D    = H @ np.diag(self.users_power) @ B.conj().T  # (nantennas, nreflects)

        sum_grad = np.zeros(self.nreflects)
        for m in range(self.nusers):
            bm = B[:, m]
            hm = H[:, m]
            alpha_m = self.users_power[m] / (1 - self.users_power[m] * np.real(hm.conj() @ Rinv @ hm))
            # bm^H - hm^H Rinv D  ->  (nreflects,) row vector
            row = bm.conj() - hm.conj() @ Rinv @ D   # shape (nreflects,)
            # C^H Rinv hm  ->  (nreflects,) column vector
            c_col = C.conj().T @ Rinv @ hm            # shape (nreflects,)
            # Q @ c_col is element-wise (Q diagonal): diag(Q) * c_col
            qc = np.diag(Q) * c_col                   # shape (nreflects,)
            # diag(row) Q C^H Rinv hm = row * qc  (element-wise)
            vec = row * qc
            sum_grad += alpha_m * np.imag(vec)

        return (2 / np.log(2)) * np.real(sum_grad)

    def gradient_modified(self, phase):
        """Gradient of cost_function_modified w.r.t. phase."""
        R    = self.autocorrelation_matrix(phase)
        trR  = np.trace(R)
        Q    = np.diag(np.exp(1j * phase))
        B    = self.Hc_ur
        C    = self.Hc_ra
        H    = self.Hc_ua + C @ Q.conj().T @ B
        D    = H @ np.diag(self.users_power) @ B.conj().T

        sum_grad = np.zeros(self.nreflects, dtype=complex)
        for m in range(self.nusers):
            hm = H[:, m]
            bm = B[:, m]
            norm2 = np.real(hm.conj() @ hm)
            alpha_m = self.users_power[m] / (trR - self.users_power[m] * norm2)

            # Q diag(bm^H) C^H hm  ->  diag(Q) * (bm.conj() * (C^H hm))
            CThm  = C.conj().T @ hm                     # (nreflects,)
            term1 = np.diag(Q) * (bm.conj() * CThm)    # (nreflects,)
            # (hm^Hhm / trR) * diag(Q C^H D)  ->  diag(Q) * diag_of(C^HD).
            # The diagonal of C^H D is obtained directly, in O(L N)
            # operations, without forming the N x N product.
            diagCTD = np.sum(np.conj(C) * D, axis=0)    # (nreflects,)
            term2 = (norm2 / trR) * np.diag(Q) * diagCTD  # (nreflects,)

            sum_grad += alpha_m * np.imag(term1 - term2)

        return (2 / np.log(2)) * np.real(sum_grad)

    # ------------------------------------------------------------------
    # Line-search methods
    # ------------------------------------------------------------------

    def _backtracking(self, cost_f, phase, grad, d, initial, max_iterations, beta, nu):
        """Standard backtracking (Armijo) line search."""
        mu    = initial
        J_old = cost_f(phase)
        J_new = cost_f(phase + mu * d)
        it    = 0
        while (J_new - J_old > nu * mu * (d @ grad)) and (it < max_iterations):
            mu    = beta * mu
            J_new = cost_f(phase + mu * d)
            it   += 1
        return mu

    def _backtracking_adaptive(self, cost_f, phase, grad, d, mu, max_iterations, beta, nu):
        """
        Adaptive backtracking: start from mu/beta before contracting.
        Ref: Fridovich-Keil & Recht.
        """
        mu    = mu / beta
        J_old = cost_f(phase)
        J_new = cost_f(phase + mu * d)
        it    = 0
        while (J_new - J_old > nu * mu * (d @ grad)) and (it < max_iterations):
            mu    = beta * mu
            J_new = cost_f(phase + mu * d)
            it   += 1
        return mu

    def _backtracking_forward(self, cost_f, phase, grad, d, mu, max_iterations, beta, nu):
        """
        Forward-then-backward line search.
        Ref: Fridovich-Keil & Recht.
        """
        mu    = mu / beta
        J_old = cost_f(phase)
        J_new = cost_f(phase + mu * d)

        # Forward: expand until Armijo fails
        it = 0
        while (J_new - J_old < nu * mu * (d @ grad)) and (it < max_iterations):
            mu    = mu / beta
            J_new = cost_f(phase + mu * d)
            it   += 1

        # Backward: contract until Armijo holds
        it = 0
        while (J_new - J_old > nu * mu * (d @ grad)) and (it < max_iterations):
            mu    = beta * mu
            J_new = cost_f(phase + mu * d)
            it   += 1
        return mu

    def _backtracking_two_way(self, cost_f, phase, grad, d, mu, initial, max_iterations, beta, nu):
        """
        Two-way backtracking (contract OR expand depending on initial check).
        Ref: Truong & Nguyen.
        """
        J_old = cost_f(phase)
        J_new = cost_f(phase + mu * d)

        if J_new - J_old > nu * mu * (d @ grad):
            return self._backtracking(cost_f, phase, grad, d, mu, max_iterations, beta, nu)

        it = 0
        while (J_new - J_old <= nu * mu * (d @ grad)) and (mu / beta <= initial) and (it < max_iterations):
            mu    = mu / beta
            J_new = cost_f(phase + mu * d)
            it   += 1
        return mu

    # ------------------------------------------------------------------
    # Quasi-Newton Hessian approximation updates
    # ------------------------------------------------------------------

    def _bfgs(self, mu, d, D, grad, grad_new):
        """
        Standard BFGS inverse Hessian update.
        Ref: Nocedal & Wright, "Numerical Optimization", p. 140.
        """
        g   = grad_new - grad
        tau = 1.0 / (mu * (g @ d))
        I   = np.eye(self.nreflects)
        D   = (I - tau * mu * np.outer(d, g)) @ D @ (I - tau * mu * np.outer(g, d)) \
              + tau * (mu ** 2) * np.outer(d, d)
        return D

    def _damped_bfgs(self, mu, d, D, grad, grad_new):
        """
        Damped BFGS inverse Hessian update.
        Ref: Nocedal & Wright, "Numerical Optimization", p. 537.
        """
        eta = 0.8
        dg  = d @ grad_new

        if dg >= eta * (d @ grad):
            xi = 1.0
        else:
            xi = eta * (d @ grad) / dg

        g   = xi * (grad_new - grad) - mu * (1 - xi) * grad
        tau = 1.0 / (mu * (g @ d))
        I   = np.eye(self.nreflects)
        D   = (I - tau * mu * np.outer(d, g)) @ D @ (I - tau * mu * np.outer(g, d)) \
              + tau * (mu ** 2) * np.outer(d, d)
        return D

    # ------------------------------------------------------------------
    # Main optimization loop
    # ------------------------------------------------------------------

    def optimize(self,
                 initial_stepsize=1.0,
                 max_iterations=1000,
                 tolerance=0.0,
                 algorithm='standard',
                 direction_method='steepest_descent',
                 ls_method='fixed',
                 ls_iterations=2000,
                 ls_beta=0.5,
                 ls_nu=1e-3,
                 verbose=True):
        """
        Optimize RIS phases for sum-rate maximization.

        Parameters
        ----------
        initial_stepsize  : initial step size (float)
        max_iterations    : maximum number of gradient iterations
        tolerance         : stopping criterion on ||grad|| / ||grad_0||
        algorithm         : 'standard' | 'modified'
        direction_method  : 'steepest_descent' | 'quasi_newton' | 'damped_bfgs'
        ls_method         : 'fixed' | 'backtracking' | 'backtracking_adaptive' |
                            'backtracking_forward' | 'backtracking_two_way'
        ls_iterations     : max line-search iterations
        ls_beta           : line-search contraction factor
        ls_nu             : Armijo sufficient decrease constant
        verbose           : print progress if True

        Returns
        -------
        W        : (nantennas, nusers) MMSE beamforming matrix
        phase    : (nreflects,) optimized phase vector [rad]
        sum_rate : (k,) sum-rate at each iteration
        """
        # Select cost / gradient pair
        if algorithm == 'standard':
            grad_func = self.gradient
            cost_f    = self.cost_function
        elif algorithm == 'modified':
            grad_func = self.gradient_modified
            cost_f    = self.cost_function_modified
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        mu    = initial_stepsize
        k     = 0
        D     = np.eye(self.nreflects)
        phase = np.zeros(self.nreflects)
        grad  = grad_func(phase)
        scale = np.linalg.norm(grad)

        sum_rate_hist = []

        while (np.linalg.norm(grad) / scale > tolerance) and (k < max_iterations):
            d = -D @ grad

            # Line search
            if ls_method == 'backtracking':
                mu = self._backtracking(cost_f, phase, grad, d, initial_stepsize, ls_iterations, ls_beta, ls_nu)
            elif ls_method == 'backtracking_adaptive':
                mu = self._backtracking_adaptive(cost_f, phase, grad, d, mu, ls_iterations, ls_beta, ls_nu)
            elif ls_method == 'backtracking_forward':
                mu = self._backtracking_forward(cost_f, phase, grad, d, mu, ls_iterations, ls_beta, ls_nu)
            elif ls_method == 'backtracking_two_way':
                mu = self._backtracking_two_way(cost_f, phase, grad, d, mu, initial_stepsize, ls_iterations, ls_beta, ls_nu)
            elif ls_method == 'fixed':
                mu = initial_stepsize
            else:
                raise ValueError(f"Unknown ls_method: {ls_method}")

            cost = self.cost_function(phase)

            if verbose:
                print(f"Iteration {k} of {max_iterations}; "
                      f"stop criterion {np.linalg.norm(grad)/scale:.6f} of {tolerance}.")
                print(f"Step-size: {mu:.6f}.")
                print(f"Sum-rate capacity: {-cost:.6f}.")

            sum_rate_hist.append(-cost)

            phase_new = phase + mu * d
            grad_new  = grad_func(phase_new)

            # Direction update
            if direction_method == 'steepest_descent':
                D = np.eye(self.nreflects)
            elif direction_method == 'quasi_newton':
                D = self._bfgs(mu, d, D, grad, grad_new)
            elif direction_method == 'damped_bfgs':
                D = self._damped_bfgs(mu, d, D, grad, grad_new)
            else:
                raise ValueError(f"Unknown direction_method: {direction_method}")

            k     += 1
            phase  = phase_new
            grad   = grad_new

        sum_rate_arr = np.array(sum_rate_hist)

        # Compute MMSE receive beamforming vectors
        R    = self.autocorrelation_matrix(phase)
        Rinv = np.linalg.pinv(R)
        Phi  = np.diag(np.exp(-1j * phase))
        W    = np.zeros((self.nantennas, self.nusers), dtype=complex)
        for m in range(self.nusers):
            hm          = self.Hc_ua[:, m] + self.Hc_ra @ Phi @ self.Hc_ur[:, m]
            W[:, m]     = Rinv @ hm / (hm.conj() @ Rinv @ hm)

        return W, phase, sum_rate_arr