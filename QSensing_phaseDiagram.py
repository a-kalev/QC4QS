#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 10:50:52 2026

@author: amirk
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hankel, svd, eig
import cvxpy as cp
from scipy.optimize import curve_fit


####################
# CORE FUNCTIONS
####################

def generate_noisy_data(amplitudes, frequencies, gamma):
    """
    Same noise model used in the current paper scripts.

    ideal:
        G(t) = cos(n t)

    shot_gnal:
        Gaussian finite-sampling perturbation around ideal.

    deco_gnal:
        mixture of shot_gnal with an additional Gaussian imperfection term.
    """
    ideal = np.cos(amplitudes * frequencies)

    sigma_shot = (ideal**2) * (np.full(len(frequencies), 1.0) - ideal**2) / 50
    shot_gnal = np.random.normal(ideal, sigma_shot)

    deco_gnal = (
        (1 - gamma) * shot_gnal
        + gamma * np.random.normal(0, 0.25, len(frequencies))
    )

    return ideal, shot_gnal, deco_gnal


def fit_alpha(G_vals, amplitudes, frequencies):
    """
    Fit the signal to cos(nfit * t), as in the existing paper scripts.
    """
    def model(frequencies, nfit):
        return np.cos(nfit * frequencies)

    popt, _ = curve_fit(model, frequencies, G_vals, p0=[0.8 * amplitudes])
    return popt[0]


def compressed_sensing(G_vals, ep):
    """
    PSD Toeplitz reconstruction, same structure as the current scripts.
    """
    len_t = len(G_vals)
    g_var = cp.Variable(len_t)

    T_entries = [[g_var[abs(i - j)] for j in range(len_t)] for i in range(len_t)]
    T_var = cp.bmat(T_entries)

    objective = cp.Minimize(cp.trace(T_var))
    constraints = [T_var >> 0, cp.norm(g_var - G_vals, 1) <= ep]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CVXOPT)

    g_opt = g_var.value
    if g_opt is None:
        raise ValueError("Optimization failed. Try increasing ep or checking input.")

    if np.abs(g_opt[0]) < 1e-12:
        raise ValueError("Optimization returned near-zero normalization.")

    g_opt = g_opt / g_opt[0]
    return np.array(g_opt)


def matrix_pencil_freq(y, dt, r=2, L=None):
    """
    Estimate angular frequencies from a sum of exponentials.
    Same matrix-pencil structure as the current scripts.
    """
    y = np.asarray(y, dtype=np.complex128).ravel()
    N = len(y)

    if L is None:
        L = N // 2

    if not (r < L < N - r + 1):
        raise ValueError("Need r < L < N-r+1. Try a larger N or different L.")

    Y0 = hankel(y[:L], y[L - 1:N - 1])
    Y1 = hankel(y[1:L + 1], y[L:N])

    U, _, Vh = svd(Y0, full_matrices=False)
    Ur = U[:, :r]
    Vr = Vh[:r, :].conj().T

    Y0r = Ur.conj().T @ Y0 @ Vr
    Y1r = Ur.conj().T @ Y1 @ Vr

    z, _ = eig(Y1r, Y0r)
    omegas = np.angle(z) / dt

    return omegas, z


def estimate_single_cosine_frequency(y, dt, r=2):
    omegas, z = matrix_pencil_freq(y, dt, r=r)
    omega = np.max(np.abs(omegas))
    return omega, omegas, z


def recon_mpen_signal(t, y, omega):
    """
    Reconstruct signal after matrix-pencil frequency estimation.
    """
    A = np.column_stack([np.cos(omega * t), np.sin(omega * t)])
    coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    a, b = coeffs
    return a * np.cos(omega * t) + b * np.sin(omega * t)


####################
# SMALL UTILITIES
####################

def center_to_edges(x):
    """
    Convert a 1D array of bin centers to bin edges for pcolormesh.
    """
    x = np.asarray(x, dtype=float)
    if len(x) == 1:
        return np.array([x[0] - 0.5, x[0] + 0.5])

    dx = np.diff(x)
    edges = np.zeros(len(x) + 1)
    edges[1:-1] = 0.5 * (x[:-1] + x[1:])
    edges[0] = x[0] - 0.5 * dx[0]
    edges[-1] = x[-1] + 0.5 * dx[-1]
    return edges


def mean_abs_relative_error(err_array, nqubit):
    """
    Use mean absolute relative error over trials:
        < |nfit - n| / n >

    This avoids cancellation between positive and negative errors.
    """
    return np.abs(np.nanmean(err_array, axis=-1)) / nqubit


####################
# MAIN ANALYSIS
####################



# Phase-diagram parameters
n_values = [50]
K_list = np.array([6, 8, 10, 12, 14, 16], dtype=int)
# Reproducibility. Set to None if you want fresh random data each run.
random_seed = n_values[0]
if random_seed is not None:
    np.random.seed(random_seed)
    
# Imperfection-noise strength gamma.
gamma_list = np.linspace(0.05, 0.3, 6)

alpha_idx = 50
R_trials = 25

# Numerical tolerance for CS.

ep_prefactor = np.array([0.1, 0.09, 0.2, 0.25, 0.25, 0.3], dtype=float)
# Storage:
# shape = (n, gamma, K)
cs_error = np.full((len(n_values), len(gamma_list), len(K_list)), np.nan)
noisy_error = np.full_like(cs_error, np.nan)
mpen_error = np.full_like(cs_error, np.nan)
shot_error = np.full_like(cs_error, np.nan)

for n_idx, nqubit in enumerate(n_values):
    print(f"\nProcessing n={nqubit} ({n_idx + 1}/{len(n_values)})")

    alpha_true_long = np.linspace(0, 2 * np.pi / nqubit, alpha_idx, dtype=float)

    for g_idx, gamma in enumerate(gamma_list):
        print(f"  gamma={gamma:.3f} ({g_idx + 1}/{len(gamma_list)})")

        for k_idx, K in enumerate(K_list):
            sample_idx = np.linspace(0, alpha_idx - 1, K, dtype=int)
            alpha_true = alpha_true_long[sample_idx]
            dt = alpha_true[1] - alpha_true[0]
            ep = ep_prefactor[k_idx]#0.25#ep_prefactor * K

            err_shot = np.full(R_trials, np.nan)
            err_noisy = np.full(R_trials, np.nan)
            err_cs = np.full(R_trials, np.nan)
            err_mpen = np.full(R_trials, np.nan)

            for r in range(R_trials):
                clean_signal, shot_signal, noisy_signal = generate_noisy_data(
                    nqubit,
                    alpha_true_long,
                    gamma
                )

                shot_samp = shot_signal[sample_idx]
                noisy_samp = noisy_signal[sample_idx]

                try:
                    err_shot[r] = fit_alpha(shot_samp, nqubit, alpha_true) - nqubit
                except Exception:
                    err_shot[r] = np.nan

                try:
                    err_noisy[r] = fit_alpha(noisy_samp, nqubit, alpha_true) - nqubit
                except Exception:
                    err_noisy[r] = np.nan

                try:
                    denoise_signal = compressed_sensing(noisy_samp, ep)
                    err_cs[r] = fit_alpha(denoise_signal, nqubit, alpha_true) - nqubit
                except Exception:
                    err_cs[r] = np.nan

                try:
                    omega_hat, _, _ = estimate_single_cosine_frequency(noisy_samp, dt, r=2)
                    mpen_signal = recon_mpen_signal(alpha_true, noisy_samp, omega_hat)
                    err_mpen[r] = fit_alpha(mpen_signal, nqubit, alpha_true) - nqubit
                except Exception:
                    err_mpen[r] = np.nan

            shot_error[n_idx, g_idx, k_idx] = mean_abs_relative_error(err_shot, nqubit)
            noisy_error[n_idx, g_idx, k_idx] = mean_abs_relative_error(err_noisy, nqubit)
            cs_error[n_idx, g_idx, k_idx] = mean_abs_relative_error(err_cs, nqubit)
            mpen_error[n_idx, g_idx, k_idx] = mean_abs_relative_error(err_mpen, nqubit)


####################
# PHASE-DIAGRAM METRICS
####################

# Absolute CS error:
#   delta alpha_CS = < |nfit_CS - n| / n >
absolute_cs_error = cs_error

# Strict usefulness relative to the best unconstrained baseline:
best_baseline_error = mpen_error

# Improvement factor:
#   > 1 means CS improves over the best baseline.
#   log10 > 0 means useful region.
eps_floor = 1e-12
improvement_factor = best_baseline_error / cs_error
log10_improvement = np.log10(improvement_factor)


####################
# PRINT SUMMARY
####################

print("\nSummary: mean absolute relative errors")
for n_idx, nqubit in enumerate(n_values):
    print(f"\nn={nqubit}")
    for g_idx, gamma in enumerate(gamma_list):
        row = []
        for k_idx, K in enumerate(K_list):
            row.append(
                f"K={K}: CS={absolute_cs_error[n_idx, g_idx, k_idx]:.3e}, "
                f"best/base={best_baseline_error[n_idx, g_idx, k_idx]:.3e}, "
                f"log10 improv={log10_improvement[n_idx, g_idx, k_idx]:+.2f}"
            )
        print(f"  gamma={gamma:.3f} | " + " | ".join(row))


####################
# PLOT: PHASE DIAGRAM OF USEFULNESS
####################

K_edges = center_to_edges(K_list)
gamma_edges = center_to_edges(gamma_list)

fig, axes = plt.subplots(
    nrows=len(n_values),
    ncols=1,
    figsize=(6, 3.0 * len(n_values)),
    sharex=True,
    sharey=True
)

axes = np.atleast_1d(axes)

# Symmetric color range around zero:
# green  -> CS better
# red    -> baseline better
# yellow -> tie
max_improve_abs = np.nanmax(np.abs(log10_improvement))
vlim_improve = max(0.25, max_improve_abs)

for n_idx, nqubit in enumerate(n_values):

    ax = axes[n_idx]

    pcm = ax.pcolormesh(
        K_edges,
        gamma_edges,
        log10_improvement[n_idx],
        shading='auto',
        cmap='RdYlGn',      # green=CS helps, red=CS hurts
        vmin=-vlim_improve,
        vmax=vlim_improve
    )

    
    ax.set_xticks(K_list,fontsize=14)
    ax.set_yticks(fontsize=14)
    ax.grid(False)




plt.tight_layout()
# plt.savefig("PhaseDiagram_Usefulness.pdf", bbox_inches="tight")
plt.show()