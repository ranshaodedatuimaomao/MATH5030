"""
1D BSDE-CFFT solver with boundary error control (Gao & Hyndman, 2025).
Baseline: Black-Scholes European call option pricing.

Reference: Algorithm 1 from the paper.
"""

import numpy as np
from scipy.stats import norm


# ─────────────────────────────────────────────
#  Black-Scholes analytic price & delta
# ─────────────────────────────────────────────

def bs_call_price(S0, K, r, sigma, T):
    """Black-Scholes European call price."""
    x = np.log(S0 / K)
    d1 = (x + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_call_delta(S0, K, r, sigma, T):
    """Black-Scholes delta."""
    x = np.log(S0 / K)
    d1 = (x + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)


# ─────────────────────────────────────────────
#  1D BSDE-CFFT solver
# ─────────────────────────────────────────────

class BSDECFFT1D:
    """
    Solves a 1D BSDE via the convolution-FFT method with boundary error control.

    State variable: X_t = log(S_t)  (log-price under risk-neutral measure)

    Forward SDE:  dX = (r - 0.5*sigma^2) dt + sigma dW
    Backward SDE: dY = -f(t,X,Y,Z) dt + Z dW,   Y_T = g(X_T)

    For a European call under Black-Scholes:
        g(x) = (exp(x) - K)^+
        f = -rY (risk-neutral discounting)
    """

    def __init__(self, r, mu, sigma, K, T, L, N, n_steps, alpha=-3.0):
        """
        Parameters
        ----------
        r      : risk-free rate
        mu     : real-world drift (used for BSDE driver; set = r for European)
        sigma  : volatility
        K      : strike
        T      : maturity
        L      : domain half-width  (grid is X0 + [-L/2, L/2])
        N      : number of spatial grid points (should be power of 2)
        n_steps: number of time steps
        alpha  : damping parameter (< 0)
        """
        self.r = r
        self.mu = mu
        self.sigma = sigma
        self.K = K
        self.T = T
        self.L = L
        self.N = N
        self.n_steps = n_steps
        self.alpha = alpha

        self.dt = T / n_steps
        self.dx = L / N
        self.dv = 2 * np.pi / L

        # Spatial grid centred at log(K)
        x0 = np.log(K) - L / 2
        self.x_grid = x0 + np.arange(N) * self.dx  # shape (N,)

        # Frequency grid (centred)
        self.v_grid = (np.arange(N) - N // 2) * self.dv  # shape (N,)

        # Phase factors for centred FFT
        self.phase = (-1) ** np.arange(N)  # (-1)^n

        # Precompute Fourier multipliers (fixed alpha, updated each step via shifting)
        self._precompute_multipliers()

    # ── characteristic function of one-step Gaussian increment ──────────────

    def _char_func(self, v):
        """ψ(v) = exp(Δt*(η*iv - 0.5*σ²*v²))"""
        eta = self.mu - 0.5 * self.sigma**2
        return np.exp(self.dt * (eta * 1j * v - 0.5 * self.sigma**2 * v**2))

    def _char_func_deriv(self, v):
        """ψ'(v) = Δt*(η*i - σ²*v) * ψ(v)"""
        eta = self.mu - 0.5 * self.sigma**2
        return self.dt * (eta * 1j - self.sigma**2 * v) * self._char_func(v)

    def _precompute_multipliers(self):
        """Ψ_y and Ψ_z evaluated at complex-shifted frequencies."""
        alpha = self.alpha
        v = self.v_grid
        psi_shifted = self._char_func(v + alpha * 1j)      # ψ(v + αi)
        self.Psi_y = psi_shifted                            # (N,)
        self.Psi_z = self.sigma * (1j * v - alpha) * psi_shifted  # (N,)

        # Recovery terms (precomputed once)
        x = self.x_grid
        psi_mi = self._char_func(-1j)           # ψ(-i), scalar
        psi_d_mi = self._char_func_deriv(-1j)   # ψ'(-i), scalar
        self.Y_ddot = np.exp(x) * psi_mi        # ..Y(x)
        self.Z_ddot = -(
            self.dt * np.exp(x) * psi_mi
            + 1j * np.exp(x) * psi_d_mi
        ) / (self.sigma * self.dt)              # ..Z(x), real part used

    # ── shifting parameter computation ───────────────────────────────────────

    def _compute_shift(self, Y):
        """
        Solve for A, B such that ũ = e^{αx}(Y - A*e^x - B) satisfies
        periodicity: ũ(x0) = ũ(xN-1)  and  ũ'(x0) = ũ'(xN-1).

        Uses one-sided finite differences for derivatives.
        """
        x = self.x_grid
        dx = self.dx
        alpha = self.alpha

        # Boundary values
        y0, yN = Y[0], Y[-1]
        # One-sided derivative estimates
        dy0 = (-3*Y[0] + 4*Y[1] - Y[2]) / (2*dx)
        dyN = (3*Y[-1] - 4*Y[-2] + Y[-3]) / (2*dx)

        x0, xN = x[0], x[-1]

        # ũ(x) = e^{αx}(Y(x) - A e^x - B)
        # ũ'(x) = α e^{αx}(Y - A e^x - B) + e^{αx}(Y' - A e^x)
        # Enforce ũ(x0) = ũ(xN) and ũ'(x0) = ũ'(xN)
        #
        # After algebra, the 2x2 linear system in A, B is:
        # [e^{x0}-e^{xN},  1-1 ]  (from ũ values)
        # [(α+1)e^{x0}-(α+1)e^{xN}, 0 ]  (from ũ' ... simplified)
        #
        # More careful derivation:
        # ũ(x0) = ũ(xN)  =>  e^{αx0}(y0 - A*e^{x0} - B) = e^{αxN}(yN - A*e^{xN} - B)
        # ũ'(x0) = ũ'(xN) =>
        #   e^{αx0}[(α y0 - A(α+1)e^{x0} - αB) + dy0]
        # = e^{αxN}[(α yN - A(α+1)e^{xN} - αB) + dyN]

        eax0 = np.exp(alpha * x0)
        eaxN = np.exp(alpha * xN)
        ex0 = np.exp(x0)
        exN = np.exp(xN)

        # System: M @ [A, B]^T = rhs
        M = np.array([
            [eax0 * ex0 - eaxN * exN,   eax0 - eaxN],
            [eax0 * (alpha + 1) * ex0 - eaxN * (alpha + 1) * exN,
             eax0 * alpha - eaxN * alpha]
        ])
        rhs = np.array([
            eax0 * y0 - eaxN * yN,
            eax0 * (alpha * y0 + dy0) - eaxN * (alpha * yN + dyN)
        ])

        try:
            AB = np.linalg.solve(M, rhs)
            A, B = AB[0].real, AB[1].real
        except np.linalg.LinAlgError:
            A, B = 0.0, 0.0

        return A, B

    # ── one backward step ───────────────────────────────────────────────────

    def _backward_step(self, Y_next):
        """Apply one backward FFT step to compute (Ŷ, Ẑ) from Y_{t+1}."""
        alpha = self.alpha
        x = self.x_grid

        # Compute shifting parameters
        A, B = self._compute_shift(Y_next)

        # Damped-shifted function
        Y_tilde = np.exp(alpha * x) * (Y_next - A * np.exp(x) - B)

        # Centred FFT
        Y_hat = np.fft.fft(self.phase * Y_tilde)   # D[(-1)^n * Y_tilde]

        # Multiply by Fourier multipliers
        Yhat_Y = Y_hat * self.Psi_y
        Yhat_Z = Y_hat * self.Psi_z

        # Inverse FFT + uncentre
        Y_dot = self.phase * np.fft.ifft(Yhat_Y).real
        Z_dot = self.phase * np.fft.ifft(Yhat_Z).real

        # Undamp and unshift
        Y_bar = np.exp(-alpha * x) * Y_dot + A * self.Y_ddot.real + B
        Z_bar = np.exp(-alpha * x) * Z_dot + A * self.Z_ddot.real

        return Y_bar, Z_bar

    # ── full backward solve ─────────────────────────────────────────────────

    def solve(self, driver_f=None, discount=True):
        """
        Run the full backward time-stepping.

        Parameters
        ----------
        driver_f : callable f(t, x, y, z) -> scalar or array, or None (zero driver)
        discount : if True and driver_f is None, use the pricing driver -rY

        Returns
        -------
        Y : array (N,)  option prices on x_grid at t=0
        Z : array (N,)  hedging ratio integrand on x_grid at t=0
        """
        # Terminal condition g(x) = (e^x - K)^+
        x = self.x_grid
        Y = np.maximum(np.exp(x) - self.K, 0.0)

        # Time grid
        times = np.linspace(self.T, 0.0, self.n_steps + 1)

        for step in range(self.n_steps):
            t_k = times[step + 1]   # current time (going backward)
            Y_next = Y.copy()

            # FFT convolution step
            Y_hat, Z_hat = self._backward_step(Y_next)

            # Driver update: Y = Ŷ + f(t, x, Ŷ, Ẑ)*Δt
            if driver_f is None and discount:
                f_val = -self.r * Y_hat
            elif driver_f is not None:
                f_val = driver_f(t_k, x, Y_hat, Z_hat)
            else:
                f_val = 0.0
            Y = Y_hat + f_val * self.dt

            # No negativity constraint for European call (Y >= 0 always)
            Y = np.maximum(Y, 0.0)

        return Y, Z_hat  # Z_hat from last step

    # ── pricing convenience ─────────────────────────────────────────────────

    def price_at(self, S0):
        """Price and delta at a given spot price S0."""
        Y, Z = self.solve()
        x_target = np.log(S0)
        price = np.interp(x_target, self.x_grid, Y)
        # Delta = Z / (sigma * S)
        delta = np.interp(x_target, self.x_grid, Z) / (self.sigma * S0)
        return price, delta


# ─────────────────────────────────────────────
#  Convergence test
# ─────────────────────────────────────────────

def run_bs_convergence_test():
    """Compare BSDE-CFFT prices vs Black-Scholes analytic formula."""
    import pandas as pd

    S0, K, r, sigma, T = 100.0, 100.0, 0.01, 0.2, 1.0
    L, alpha = 10.0, -3.0

    results = []
    for N_exp in [10, 11, 12]:
        N = 2**N_exp
        for n_steps in [1000, 2000]:
            solver = BSDECFFT1D(
                r=r, mu=r, sigma=sigma, K=K, T=T,
                L=L, N=N, n_steps=n_steps, alpha=alpha
            )
            price_num, delta_num = solver.price_at(S0)
            price_bs = bs_call_price(S0, K, r, sigma, T)
            delta_bs = bs_call_delta(S0, K, r, sigma, T)
            results.append({
                'N': N, 'n_steps': n_steps,
                'Price_CFFT': price_num,
                'Price_BS': price_bs,
                '|err_price|': abs(price_num - price_bs),
                'Delta_CFFT': delta_num,
                'Delta_BS': delta_bs,
                '|err_delta|': abs(delta_num - delta_bs),
            })
            print(f"N=2^{N_exp}, n={n_steps}: "
                  f"price={price_num:.6f} (BS={price_bs:.6f}), "
                  f"delta={delta_num:.6f} (BS={delta_bs:.6f})")

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("1D BSDE-CFFT  –  Black-Scholes European Call")
    print("=" * 60)
    run_bs_convergence_test()
