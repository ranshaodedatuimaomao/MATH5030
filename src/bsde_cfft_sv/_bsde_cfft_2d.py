"""
2D local-kernel CFFT pricing prototype with boundary error control.

State: (X_t, V_t) = (log S_t, variance_t)
This is a Y/Z extension inspired by the 1D boundary-controlled
CFFT method (Gao & Hyndman, 2025). The local-kernel path computes prices and
Markovian BSDE controls Z = grad(u) * diffusion. Fully nonlinear Z-dependent
drivers still require a separate convergence study.

Key ideas for the 2D extension:
  - Replace 1D FFT convolution with batched 2D FFT convolution on a tensor grid.
  - The transition density of (X_{t+Δt}, V_{t+Δt}) | (X_t, V_t) is
    approximated by a short-time 2D Gaussian (Euler-Maruyama).
  - Damping is applied independently in each dimension:
      ũ(x,v) = e^{α_x * x + α_v * v} * (u(x,v) - h(x,v))
  - The shifting function h(x,v) is chosen to enforce periodicity
    on the 2D boundary.
  - State-dependent coefficients are handled by one local Gaussian kernel per
    current variance slice. This is a local-kernel FFT approximation, not one
    globally translation-invariant convolution.

Models implemented:
  - Heston (benchmark, closed-form available via characteristic function)
  - GARCH Diffusion (main application, no closed-form FFT pricing)
"""

import os
import numpy as np
from scipy import fft as sp_fft



class TwoDimDensity:
    """
    Short-time transition density for a 2D Itô diffusion:

        dX = η_x(X,V) dt + σ_xx(X,V) dW1 + σ_xy(X,V) dW2
        dV = η_v(X,V) dt + σ_vx(X,V) dW1 + σ_vv(X,V) dW2

    The joint increment (ΔX, ΔV) | (X,V) = (x,v) is approximately N(μ, Σ) where
        μ = (η_x Δt, η_v Δt)
        Σ = A A^T Δt,   A = [[σ_xx, σ_xy],[σ_vx, σ_vv]]

    The 2D characteristic function of the increment is:
        ψ(ξ_x, ξ_v) = exp(Δt * [iμ·ξ - ½ ξ^T Σ ξ])
    """

    def __init__(self, mu_x, mu_v, Sigma, dt):
        """
        mu_x, mu_v : floats – drift coefficients times Δt
        Sigma      : 2x2 array – diffusion covariance matrix (= A A^T)
        dt         : float – time step
        """
        self.mu_x = mu_x
        self.mu_v = mu_v
        self.Sigma = Sigma   # 2x2
        self.dt = dt

    def char_func_2d(self, xi_x, xi_v):
        """
        2D characteristic function of the short-time increment.
        xi_x, xi_v: frequency variables (arrays broadcastable)
        Returns complex array of same shape.
        """
        i = 1j
        drift_term = i * (self.mu_x * xi_x + self.mu_v * xi_v) * self.dt
        # Quadratic form: ξ^T Σ ξ * Δt
        S = self.Sigma
        quad = (S[0,0] * xi_x**2
                + (S[0,1] + S[1,0]) * xi_x * xi_v
                + S[1,1] * xi_v**2) * self.dt
        return np.exp(drift_term - 0.5 * quad)




class BSDECFFT2D:
    """
    Generic 2D local-kernel CFFT pricing prototype with boundary control.

    The two state variables are (X, V): log-price and variance (or vol proxy).

    Grid:
        x in [x0, x0+Lx],  Nx points
        v in [v0, v0+Lv],  Nv points (v > 0 enforced by grid choice)

    Damping: independent α_x < 0 for X, α_v for V (positive or negative).
    Shifting: h(x,v) = A(v) * exp(x) + B(v), with an optional discrete
    Neumann control on the variance boundary.
    Zx/Zv denote controls with respect to the two independent Brownian drivers.
    """

    def __init__(self, Nx, Nv, Lx, Lv, x0, v0, n_steps, T,
                 alpha_x=-3.0, alpha_v=0.0, v_boundary="neumann",
                 fft_workers=None):
        self.Nx = Nx
        self.Nv = Nv
        self.Lx = Lx
        self.Lv = Lv
        self.x0 = x0
        self.v0 = v0
        self.n_steps = n_steps
        self.T = T
        self.alpha_x = alpha_x
        self.alpha_v = alpha_v
        self.v_boundary = v_boundary
        self.use_v_shift = False
        self.driver_depends_on_z = False
        self.fft_workers = self._resolve_fft_workers(fft_workers)

        self.dt = T / n_steps
        self.dx = Lx / Nx
        self.dv_grid = Lv / Nv
        self.dvx = 2 * np.pi / Lx   # frequency step in x
        self.dvv = 2 * np.pi / Lv   # frequency step in v

        # Spatial grid
        self.x_grid = x0 + np.arange(Nx) * self.dx        # (Nx,)
        self.v_vals = v0 + np.arange(Nv) * self.dv_grid   # (Nv,)
        self.XX, self.VV = np.meshgrid(self.x_grid, self.v_vals, indexing='ij')
        # XX, VV : (Nx, Nv)
        self.expX = np.exp(self.XX)
        self.damp_grid = np.exp(self.alpha_x * self.XX + self.alpha_v * self.VV)
        self.undamp_grid = 1.0 / self.damp_grid

        # Frequency grid
        self.vx_grid = (np.arange(Nx) - Nx//2) * self.dvx
        self.vv_grid = (np.arange(Nv) - Nv//2) * self.dvv
        self.VX, self.VVV = np.meshgrid(self.vx_grid, self.vv_grid, indexing='ij')
        # VX, VVV : (Nx, Nv)

        # Phase factors for 2D centring
        nx_arr = np.arange(Nx)
        nv_arr = np.arange(Nv)
        NX, NV = np.meshgrid(nx_arr, nv_arr, indexing='ij')
        self.phase2d = (-1) ** (NX + NV)   # (Nx, Nv)
        self._diag_cols = np.arange(Nv)
        self._precompute_x_shift_system()

    @staticmethod
    def _resolve_fft_workers(fft_workers):
        if fft_workers is not None:
            return int(fft_workers)
        env_workers = os.getenv("CFFT_FFT_WORKERS")
        if env_workers:
            return int(env_workers)
        return min(6, os.cpu_count() or 1)

    def _precompute_x_shift_system(self):
        alpha = self.alpha_x
        x = self.x_grid
        x0, xN = x[0], x[-1]
        eax0 = np.exp(alpha * x0)
        eaxN = np.exp(alpha * xN)
        ex0 = np.exp(x0)
        exN = np.exp(xN)
        M = np.array([
            [eax0 * ex0 - eaxN * exN,   eax0 - eaxN],
            [eax0 * (alpha+1)*ex0 - eaxN * (alpha+1)*exN,
             eax0 * alpha - eaxN * alpha]
        ])
        self._x_shift_M = M
        self._x_shift_constants = (eax0, eaxN)

    # ── Methods to override in subclasses ───────────────────────────────────

    def char_func_2d(self, xi_x, xi_v, x, v):
        """
        Return the 2D characteristic function of (ΔX, ΔV) | (X,V)=(x,v).
        By default evaluates at representative (x̄, v̄) = grid centre.
        Override per model for spatially-varying coefficients.
        """
        raise NotImplementedError

    def terminal_payoff(self, XX, VV):
        """Terminal condition g(x,v). Override per model/payoff."""
        raise NotImplementedError

    def driver_f(self, t, XX, VV, Y, Z_x, Z_v):
        """Default risk-neutral pricing driver."""
        r = getattr(self, "r", 0.0)
        return -r * Y

    # ── Shifting (2D) ────────────────────────────────────────────────────────

    def _compute_shift_2d(self, Y):
        """
        Compute shifting parameters A(v), B(v) for each variance slice.

        For each v-slice, Y(·, v) is a 1D function of x.
        We solve the same 2-parameter system as in 1D for each slice.
        Returns A, B arrays of shape (Nv,).
        """
        dx = self.dx
        alpha = self.alpha_x
        eax0, eaxN = self._x_shift_constants

        y0 = Y[0, :]
        yN = Y[-1, :]
        dy0 = (-3*Y[0, :] + 4*Y[1, :] - Y[2, :]) / (2*dx)
        dyN = (3*Y[-1, :] - 4*Y[-2, :] + Y[-3, :]) / (2*dx)
        rhs = np.vstack([
            eax0 * y0 - eaxN * yN,
            eax0 * (alpha * y0 + dy0) - eaxN * (alpha * yN + dyN)
        ])
        try:
            AB = np.linalg.solve(self._x_shift_M, rhs)
            return AB[0].real, AB[1].real
        except np.linalg.LinAlgError:
            return np.zeros(self.Nv), np.zeros(self.Nv)

    def _compute_v_shift_2d(self, R):
        """
        Optional variance-direction shift C(x)*v + D(x).

        It is applied after the x-shift, on the residual R. With alpha_v != 0
        it enforces periodicity of exp(alpha_v*v)*(R - C*v - D) in value and
        first derivative along v for each x-slice. If alpha_v is zero, no
        variance shift is applied.
        """
        alpha = self.alpha_v
        if abs(alpha) < 1e-14 or not self.use_v_shift:
            return np.zeros(self.Nx), np.zeros(self.Nx)

        v = self.v_vals
        dv = self.dv_grid
        C_arr = np.zeros(self.Nx)
        D_arr = np.zeros(self.Nx)

        v0, vN = v[0], v[-1]
        eav0 = np.exp(alpha * v0)
        eavN = np.exp(alpha * vN)

        M = np.array([
            [eav0 * v0 - eavN * vN, eav0 - eavN],
            [eav0 * (alpha * v0 + 1.0) - eavN * (alpha * vN + 1.0),
             eav0 * alpha - eavN * alpha]
        ])

        for i in range(self.Nx):
            Ri = R[i, :]
            r0, rN = Ri[0], Ri[-1]
            dr0 = (-3*Ri[0] + 4*Ri[1] - Ri[2]) / (2*dv)
            drN = (3*Ri[-1] - 4*Ri[-2] + Ri[-3]) / (2*dv)
            rhs = np.array([
                eav0 * r0 - eavN * rN,
                eav0 * (alpha * r0 + dr0) - eavN * (alpha * rN + drN)
            ])
            try:
                CD = np.linalg.solve(M, rhs)
                C_arr[i], D_arr[i] = CD[0].real, CD[1].real
            except np.linalg.LinAlgError:
                C_arr[i], D_arr[i] = 0.0, 0.0

        return C_arr, D_arr

    # ── One backward step (2D FFT) ───────────────────────────────────────────

    def _backward_step_2d(self, Y_next, Psi_y, Psi_zx, Psi_zv,
                          Y_ddot, Z_ddot):
        """
        One 2D backward FFT step.

        Parameters
        ----------
        Y_next : (Nx, Nv)  –  Y at t+Δt
        Psi_y, Psi_zx, Psi_zv : (Nx, Nv) – Fourier multipliers
        Y_ddot, Z_ddot : (Nx, Nv) – recovery terms

        Returns
        -------
        Y_bar, Zx_bar, Zv_bar : (Nx, Nv)
        """
        alpha_x = self.alpha_x
        XX = self.XX

        # Shifting (along x for each v-slice)
        A_arr, B_arr = self._compute_shift_2d(Y_next)

        # Broadcast A, B: shape (1, Nv)
        A2d = A_arr[np.newaxis, :]   # (1, Nv)
        B2d = B_arr[np.newaxis, :]   # (1, Nv)

        # Damped-shifted function: ũ = e^{αx * x}(Y - A e^x - B)
        shift = A2d * self.expX + B2d
        Y_tilde = np.exp(alpha_x * XX) * (Y_next - shift)  # (Nx, Nv)

        # Centred 2D FFT
        Y_hat = sp_fft.fft2(self.phase2d * Y_tilde, workers=self.fft_workers)

        # Convolution in frequency domain (pointwise products)
        Yhat_Y  = Y_hat * Psi_y
        Yhat_Zx = Y_hat * Psi_zx
        Yhat_Zv = Y_hat * Psi_zv

        # Inverse 2D FFT + uncentre
        Y_dot  = self.phase2d * sp_fft.ifft2(Yhat_Y, workers=self.fft_workers).real
        Zx_dot = self.phase2d * sp_fft.ifft2(Yhat_Zx, workers=self.fft_workers).real
        Zv_dot = self.phase2d * sp_fft.ifft2(Yhat_Zv, workers=self.fft_workers).real

        # Undamp and unshift
        exp_neg_ax = np.exp(-alpha_x * XX)
        Y_bar  = exp_neg_ax * Y_dot  + A2d * Y_ddot + B2d
        Zx_bar = exp_neg_ax * Zx_dot + A2d * Z_ddot
        Zv_bar = exp_neg_ax * Zv_dot   # no x-shift contribution for v-component

        return Y_bar, Zx_bar, Zv_bar

    def _ifft2_diagonal(self, Y_hat, Psi):
        """
        Compute only output column j of the convolution using kernel j.

        The previous implementation performed one full 2D inverse FFT for each
        variance kernel and then kept only column j. This diagonal variant keeps
        the same result but skips the unused x-IFFTs for all other columns.
        """
        cols = self._diag_cols
        tmp_v = sp_fft.ifft(
            Y_hat[np.newaxis, :, :] * Psi, axis=2, workers=self.fft_workers
        )
        diag_v = tmp_v[cols, :, cols]
        tmp_x = sp_fft.ifft(diag_v, axis=1, workers=self.fft_workers)
        return (self.phase2d * tmp_x.T).real

    def _backward_step_2d_local(self, Y_next, multipliers, compute_z=False):
        """
        One backward step with variance-local Gaussian kernels.

        The global 2D FFT kernel is translation invariant in both x and v. For
        stochastic volatility models that is too aggressive: if the payoff is
        independent of v, a single kernel evaluated at v_bar keeps the whole
        solution almost flat in v. Here we build one kernel per current
        variance grid point v_j and retain the output column j from that
        convolution. This is still a short-time Gaussian CFFT approximation,
        but the coefficients seen by each node are model- and state-dependent.
        """
        alpha_x = self.alpha_x
        alpha_v = self.alpha_v
        VV = self.VV

        # Boundary shift along x, with one pair A(v), B(v) per v-slice.
        A_arr, B_arr = self._compute_shift_2d(Y_next)
        A2d = A_arr[np.newaxis, :]
        B2d = B_arr[np.newaxis, :]
        x_shift = A2d * self.expX + B2d

        # Optional secondary shift along v for the residual.
        R = Y_next - x_shift
        C_arr, D_arr = self._compute_v_shift_2d(R)
        C2d = C_arr[:, np.newaxis]
        D2d = D_arr[:, np.newaxis]
        v_shift = C2d * VV + D2d

        Y_tilde = self.damp_grid * (Y_next - x_shift - v_shift)

        Y_hat = sp_fft.fft2(self.phase2d * Y_tilde, workers=self.fft_workers)
        Y_dot = self._ifft2_diagonal(Y_hat, multipliers["Psi_y"])
        A_expect = multipliers["shift_A_weights"] @ A_arr
        B_expect = multipliers["shift_B_weights"] @ B_arr
        x_shift_dot = self.expX * A_expect[np.newaxis, :] + B_expect[np.newaxis, :]

        if self.use_v_shift and abs(alpha_v) > 1e-14:
            v_shift_hat = sp_fft.fft2(self.phase2d * v_shift, workers=self.fft_workers)
            v_shift_dot = self._ifft2_diagonal(v_shift_hat, multipliers["Psi_plain"])
        else:
            v_shift_dot = 0.0

        Y_bar = self.undamp_grid * Y_dot + x_shift_dot + v_shift_dot

        if compute_z:
            Zx_bar, Zv_bar = self._z_from_price_grid(Y_bar)
        else:
            Zx_bar = np.zeros_like(Y_bar)
            Zv_bar = np.zeros_like(Y_bar)
        return Y_bar, Zx_bar, Zv_bar

    def _apply_variance_boundary_control(self, Y):
        """
        Stable discrete boundary control in the variance direction.

        A full v-shifting recovery is delicate for state-dependent kernels. The
        default Neumann control prevents the FFT iteration from feeding sharp
        artificial variance-edge slopes back into the next step.
        """
        if self.v_boundary != "neumann" or self.Nv < 3:
            return Y
        Y[:, 0] = Y[:, 1]
        Y[:, -1] = Y[:, -2]
        return Y

    # ── Full backward solve ─────────────────────────────────────────────────

    def solve(self):
        """
        Run the full 2D backward iteration.
        Returns Y (Nx, Nv), Zx (Nx, Nv), Zv (Nx, Nv) at t=0.
        """
        XX, VV = self.XX, self.VV

        # Terminal condition
        Y = self.terminal_payoff(XX, VV)   # (Nx, Nv)
        Y = self._apply_variance_boundary_control(Y)

        # Precompute Fourier multipliers.
        multipliers = self._build_multipliers()
        local_multipliers = (
            isinstance(multipliers, dict) and multipliers.get("local", False)
        )
        if not local_multipliers:
            Psi_y, Psi_zx, Psi_zv, Y_ddot, Z_ddot = multipliers

        times = np.linspace(self.T, 0.0, self.n_steps + 1)
        Zx_bar = np.zeros_like(Y)
        Zv_bar = np.zeros_like(Y)

        for step in range(self.n_steps):
            t_k = times[step + 1]
            Y_next = Y.copy()

            if local_multipliers:
                Y_hat, Zx_hat, Zv_hat = self._backward_step_2d_local(
                    Y_next, multipliers, compute_z=self.driver_depends_on_z
                )
            else:
                Y_hat, Zx_hat, Zv_hat = self._backward_step_2d(
                    Y_next, Psi_y, Psi_zx, Psi_zv, Y_ddot, Z_ddot
                )

            # Driver update
            f_val = self.driver_f(t_k, XX, VV, Y_hat, Zx_hat, Zv_hat)
            Y = Y_hat + f_val * self.dt
            Y = np.maximum(Y, 0.0)
            Y = self._apply_variance_boundary_control(Y)
            if local_multipliers:
                Zx_bar, Zv_bar = self._z_from_price_grid(Y)
            else:
                Zx_bar, Zv_bar = Zx_hat, Zv_hat

        return Y, Zx_bar, Zv_bar

    def _build_multipliers(self):
        """
        Build Fourier multipliers Ψ_y, Ψ_zx, Ψ_zv and recovery terms.
        Subclasses may return either one global tuple of multipliers or a
        dict with local variance-indexed multipliers.
        """
        raise NotImplementedError

    def _interp2d(self, x_q, v_q, F):
        """Bilinear interpolation of F on (x_grid, v_grid)."""
        ix = np.searchsorted(self.x_grid, x_q) - 1
        iv = np.searchsorted(self.v_vals, v_q) - 1
        ix = np.clip(ix, 0, self.Nx - 2)
        iv = np.clip(iv, 0, self.Nv - 2)
        wx = (x_q - self.x_grid[ix]) / self.dx
        wv = (v_q - self.v_vals[iv]) / self.dv_grid
        return ((1-wx)*(1-wv)*F[ix,iv] + wx*(1-wv)*F[ix+1,iv] +
                (1-wx)*wv*F[ix,iv+1] + wx*wv*F[ix+1,iv+1])

    def _x_derivative_grid(self, F):
        """Finite-difference derivative dF/dx on the x grid."""
        dF = np.empty_like(F)
        dF[1:-1, :] = (F[2:, :] - F[:-2, :]) / (2 * self.dx)
        dF[0, :] = (-3*F[0, :] + 4*F[1, :] - F[2, :]) / (2 * self.dx)
        dF[-1, :] = (3*F[-1, :] - 4*F[-2, :] + F[-3, :]) / (2 * self.dx)
        return dF

    def _v_derivative_grid(self, F):
        """Finite-difference derivative dF/dv on the variance grid."""
        dF = np.empty_like(F)
        dF[:, 1:-1] = (F[:, 2:] - F[:, :-2]) / (2 * self.dv_grid)
        dF[:, 0] = (-3*F[:, 0] + 4*F[:, 1] - F[:, 2]) / (2 * self.dv_grid)
        dF[:, -1] = (3*F[:, -1] - 4*F[:, -2] + F[:, -3]) / (2 * self.dv_grid)
        return dF

    def _z_from_price_grid(self, Y):
        """Compute Brownian BSDE controls Z = grad(u) * diffusion."""
        u_x = self._x_derivative_grid(Y)
        u_v = self._v_derivative_grid(Y)
        return self._z_from_gradients(u_x, u_v)

    def _z_from_gradients(self, u_x, u_v):
        """Model-specific Brownian control map. Subclasses override this."""
        return np.zeros_like(u_x), np.zeros_like(u_v)

    def _delta_from_price_grid(self, S0, V0, Y):
        """Delta = dY/dS = (dY/dx) / S."""
        x0 = np.log(S0)
        dYdx = self._interp2d(x0, V0, self._x_derivative_grid(Y))
        return dYdx / S0

    def price_delta_z_at(self, S0, V0):
        """Return price, delta, and Brownian controls (Zx, Zv)."""
        Y, Zx, Zv = self.solve()
        x0 = np.log(S0)
        price = self._interp2d(x0, V0, Y)
        delta = self._delta_from_price_grid(S0, V0, Y)
        z_x = self._interp2d(x0, V0, Zx)
        z_v = self._interp2d(x0, V0, Zv)
        return price, delta, z_x, z_v

    def _shift_recovery_weights(self, v, eta_x, eta_v,
                                sigma_11, sigma_12, sigma_22):
        """
        Weights for E[A(V') exp(X') + B(V') | X=x, V=v].

        For a local Gaussian increment, B only needs the marginal transition
        weights of V'. The A exp(X') part uses the conditional Gaussian moment
        E[exp(Delta X) | Delta V].
        """
        dt = self.dt
        v_targets = self.v_vals
        mean_dv = eta_v * dt
        mean_v = v + mean_dv
        var_x = max(sigma_11 * dt, 0.0)
        var_v = max(sigma_22 * dt, 0.0)
        cov_xv = sigma_12 * dt
        mean_dx = eta_x * dt

        if var_v < 1e-14:
            weights = np.zeros(self.Nv)
            k = np.argmin(np.abs(v_targets - mean_v))
            weights[k] = 1.0
            exp_weights = weights * np.exp(mean_dx + 0.5 * var_x)
            return exp_weights, weights

        z = v_targets - mean_v
        weights = np.exp(-0.5 * z**2 / var_v)
        weights_sum = weights.sum()
        if weights_sum == 0.0:
            k = np.argmin(np.abs(v_targets - mean_v))
            weights[k] = 1.0
        else:
            weights /= weights_sum

        delta_v = v_targets - v
        cond_var_x = max(var_x - cov_xv**2 / var_v, 0.0)
        cond_mean_x = mean_dx + (cov_xv / var_v) * (delta_v - mean_dv)
        exp_weights = weights * np.exp(cond_mean_x + 0.5 * cond_var_x)
        return exp_weights, weights




class HestonBSDECFFT(BSDECFFT2D):
    """
    Heston stochastic volatility model:

        dX  = (r - ½V) dt + √V dW1
        dV  = κ(θ - V) dt + ξ √V [ρ dW1 + √(1-ρ²) dW2]

    State: X = log S, V = variance.

    Covariance matrix of increments (X, V) per unit time:
        Σ = [[V,        ρ ξ V ],
             [ρ ξ V,    ξ² V  ]]

    Since Σ depends on V, the implementation below builds one short-time
    Gaussian CFFT kernel per variance grid point.

    For the 2D shifting, we use h(x,v) = A(v) e^x + B(v) — the shift is in x only,
    matching the call payoff structure.
    """

    def __init__(self, r, kappa, theta, xi, rho, K, T,
                 Nx, Nv, Lx, Lv, n_steps,
                 v_center=None, alpha_x=-3.0, alpha_v=0.0,
                 v_boundary="neumann", fft_workers=None):
        self.r = r
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho
        self.K = K

        # Grid: x centred at log(K), v from near 0 to v_max
        x0 = np.log(K) - Lx / 2
        v0 = 1e-4  # avoid v=0 boundary issues
        super().__init__(
            Nx, Nv, Lx, Lv, x0, v0, n_steps, T, alpha_x, alpha_v,
            v_boundary, fft_workers
        )

        # Kept for backward-compatible construction; kernels now use v-grid levels.
        self.v_bar = theta if v_center is None else v_center

    def terminal_payoff(self, XX, VV):
        return np.maximum(np.exp(XX) - self.K, 0.0)

    def _build_multipliers(self):
        """
        Build 2D Fourier multipliers using local variance levels.
        """
        r, kappa, theta, xi, rho = self.r, self.kappa, self.theta, self.xi, self.rho
        dt, alpha_x, alpha_v = self.dt, self.alpha_x, self.alpha_v
        VX, VVV = self.VX, self.VVV
        Psi_y = np.empty((self.Nv, self.Nx, self.Nv), dtype=complex)
        Psi_plain = np.empty_like(Psi_y)
        shift_A_weights = np.empty((self.Nv, self.Nv))
        shift_B_weights = np.empty((self.Nv, self.Nv))

        vx_shift = VX + alpha_x * 1j
        vv_shift = VVV + alpha_v * 1j

        for j, v in enumerate(np.maximum(self.v_vals, 1e-12)):
            eta_x = r - 0.5 * v
            eta_v = kappa * (theta - v)

            sigma_11 = v
            sigma_12 = rho * xi * v
            sigma_22 = xi**2 * v

            def psi(vx, vv):
                drift = dt * (1j * eta_x * vx + 1j * eta_v * vv)
                quad = dt * (
                    sigma_11 * vx**2
                    + 2.0 * sigma_12 * vx * vv
                    + sigma_22 * vv**2
                )
                return np.exp(drift - 0.5 * quad)

            Psi_y[j] = psi(vx_shift, vv_shift)
            Psi_plain[j] = psi(VX, VVV)
            shift_A_weights[j], shift_B_weights[j] = self._shift_recovery_weights(
                v, eta_x, eta_v, sigma_11, sigma_12, sigma_22
            )

        return {
            "local": True,
            "Psi_y": Psi_y,
            "Psi_plain": Psi_plain,
            "shift_A_weights": shift_A_weights,
            "shift_B_weights": shift_B_weights,
        }

    def _z_from_gradients(self, u_x, u_v):
        v = np.maximum(self.VV, 0.0)
        sqrt_v = np.sqrt(v)
        rho_bar = np.sqrt(max(1.0 - self.rho**2, 0.0))
        z_x = sqrt_v * u_x + self.rho * self.xi * sqrt_v * u_v
        z_v = rho_bar * self.xi * sqrt_v * u_v
        return z_x, z_v

    def price_at(self, S0, V0):
        """Price and delta at given (S0, V0)."""
        price, delta, _, _ = self.price_delta_z_at(S0, V0)
        return price, delta

    def _interp2d(self, x_q, v_q, F):
        """Bilinear interpolation of F on (x_grid, v_grid)."""
        ix = np.searchsorted(self.x_grid, x_q) - 1
        iv = np.searchsorted(self.v_vals, v_q) - 1
        ix = np.clip(ix, 0, self.Nx - 2)
        iv = np.clip(iv, 0, self.Nv - 2)
        wx = (x_q - self.x_grid[ix]) / self.dx
        wv = (v_q - self.v_vals[iv]) / self.dv_grid
        return ((1-wx)*(1-wv)*F[ix,iv] + wx*(1-wv)*F[ix+1,iv] +
                (1-wx)*wv*F[ix,iv+1] + wx*wv*F[ix+1,iv+1])




class GARCHDiffusionBSDECFFT(BSDECFFT2D):
    """
    GARCH Diffusion model (Nelson 1990 / Barone-Adesi et al. 2005):

        dS/S = μ dt + σ_t dW1
        d(σ²) = a(b - σ²) dt + c σ² dW2    (GARCH diffusion on variance)

    In log-price coordinates X = log S:
        dX  = (μ - ½ σ²) dt + σ dW1
        dσ² = a(b - σ²) dt + c σ² dW2

    Equivalently with V = σ²:
        dX  = (μ - ½ V) dt + √V dW1
        dV  = a(b - V) dt + c V dW2

    Note: W1 and W2 can be correlated (ρ) or independent (ρ=0).
    For pricing, the X drift is evaluated under the risk-neutral measure
    with r in place of μ.
    Unlike Heston, the vol-of-vol term is c*V (not c*√V), so the
    model does NOT have an exact FFT characteristic function solution
    → BSDE-CFFT provides a genuine numerical alternative.

    Key difference from Heston:
        Diffusion covariance at (x,v): Σ = [[V, ρ c V^{3/2}], [ρ c V^{3/2}, c² V²]]
    """

    def __init__(self, r, mu, a, b, c, rho, K, T,
                 Nx, Nv, Lx, Lv, n_steps,
                 v_center=None, alpha_x=-3.0, alpha_v=0.0,
                 v_boundary="neumann", fft_workers=None):
        self.r = r
        self.mu = mu
        self.a = a     # mean-reversion speed
        self.b = b     # long-run variance
        self.c = c     # vol-of-vol coefficient
        self.rho = rho
        self.K = K

        x0 = np.log(K) - Lx / 2
        v0 = 1e-5
        super().__init__(
            Nx, Nv, Lx, Lv, x0, v0, n_steps, T, alpha_x, alpha_v,
            v_boundary, fft_workers
        )

        # Kept for backward-compatible construction; kernels now use v-grid levels.
        self.v_bar = b if v_center is None else v_center

    def terminal_payoff(self, XX, VV):
        return np.maximum(np.exp(XX) - self.K, 0.0)

    def _build_multipliers(self):
        """
        Build Fourier multipliers for GARCH diffusion at local variance levels.

        Covariance matrix of (ΔX, ΔV) at v̄:
            Σ_11 = v̄          (var of X)
            Σ_12 = ρ c v̄^{3/2} (cov X,V)
            Σ_22 = c² v̄²      (var of V)
        """
        r = self.r
        a, b, c, rho = self.a, self.b, self.c, self.rho
        dt, alpha_x, alpha_v = self.dt, self.alpha_x, self.alpha_v
        VX, VVV = self.VX, self.VVV
        Psi_y = np.empty((self.Nv, self.Nx, self.Nv), dtype=complex)
        Psi_plain = np.empty_like(Psi_y)
        shift_A_weights = np.empty((self.Nv, self.Nv))
        shift_B_weights = np.empty((self.Nv, self.Nv))

        vx_shift = VX + alpha_x * 1j
        vv_shift = VVV + alpha_v * 1j

        for j, v in enumerate(np.maximum(self.v_vals, 1e-12)):
            # Pricing is under the risk-neutral measure; mu is retained for
            # compatibility with the paper notation and benchmark signature.
            eta_x = r - 0.5 * v
            eta_v = a * (b - v)

            sigma_11 = v
            sigma_12 = rho * c * v**1.5
            sigma_22 = c**2 * v**2

            def psi(vx, vv):
                drift = dt * (1j * eta_x * vx + 1j * eta_v * vv)
                quad = dt * (
                    sigma_11 * vx**2
                    + 2.0 * sigma_12 * vx * vv
                    + sigma_22 * vv**2
                )
                return np.exp(drift - 0.5 * quad)

            Psi_y[j] = psi(vx_shift, vv_shift)
            Psi_plain[j] = psi(VX, VVV)
            shift_A_weights[j], shift_B_weights[j] = self._shift_recovery_weights(
                v, eta_x, eta_v, sigma_11, sigma_12, sigma_22
            )

        return {
            "local": True,
            "Psi_y": Psi_y,
            "Psi_plain": Psi_plain,
            "shift_A_weights": shift_A_weights,
            "shift_B_weights": shift_B_weights,
        }

    def _z_from_gradients(self, u_x, u_v):
        v = np.maximum(self.VV, 0.0)
        sqrt_v = np.sqrt(v)
        rho_bar = np.sqrt(max(1.0 - self.rho**2, 0.0))
        z_x = sqrt_v * u_x + self.rho * self.c * v * u_v
        z_v = rho_bar * self.c * v * u_v
        return z_x, z_v

    def price_at(self, S0, V0):
        """Price and delta at given (S0, V0=σ₀²)."""
        price, delta, _, _ = self.price_delta_z_at(S0, V0)
        return price, delta

    def _interp2d(self, x_q, v_q, F):
        ix = np.searchsorted(self.x_grid, x_q) - 1
        iv = np.searchsorted(self.v_vals, v_q) - 1
        ix = np.clip(ix, 0, self.Nx - 2)
        iv = np.clip(iv, 0, self.Nv - 2)
        wx = (x_q - self.x_grid[ix]) / self.dx
        wv = (v_q - self.v_vals[iv]) / self.dv_grid
        return ((1-wx)*(1-wv)*F[ix,iv] + wx*(1-wv)*F[ix+1,iv] +
                (1-wx)*wv*F[ix,iv+1] + wx*wv*F[ix+1,iv+1])
