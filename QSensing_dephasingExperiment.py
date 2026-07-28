"""
Created on Wed Jun  3 10:50:52 2026

@author: amirk
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz, hankel, svd, eig
import cvxpy as cp
from scipy.optimize import curve_fit


def generate_noisy_data(amplitudes, frequencies, gamma, dephasing_rate=0.0):
    """Generate an exponentially damped cosine and two noisy readouts.

    The dephasing rate is used only in the synthetic data generator; the estimators
    below are deliberately not given access to it.

    Parameters
    ----------
    amplitudes : float
        Oscillation frequency scale (in the paper script this is n).
    frequencies : array_like
        Time grid.
    gamma : float
        Additional imperfection mixing weight.
    dephasing_rate : float
        Exponential decay rate in the ideal signal.
    """
    ideal = np.exp(-dephasing_rate * frequencies) * np.cos(amplitudes * frequencies)
    sigma_shot = (ideal ** 2) * (np.full(len(frequencies), 1.0) - (ideal ** 2)) / 50

    shot_gnal = np.random.normal(ideal, sigma_shot)
    deco_gnal = (1 - gamma) * shot_gnal + gamma * np.random.normal(0, 0.25, len(frequencies))

    return ideal, shot_gnal, deco_gnal


def fit_alpha(G_vals, amplitudes, frequencies):
    # Deliberately fit to the pure cosine model; dephasing is treated as an unknown
    # nuisance effect and is not supplied to the estimator.
    def model(frequencies, nfit):
        return np.cos(nfit * frequencies)

    popt, pcov = curve_fit(model, frequencies, G_vals, p0=[0.8 * amplitudes])
    qubit_est = popt[0]
    return qubit_est


def compressed_sensing(G_vals, ep):
    len_t = len(G_vals)
    g_var = cp.Variable(len_t)

    T_entries = [[g_var[abs(i - j)] for j in range(len_t)] for i in range(len_t)]
    T_var = cp.bmat(T_entries)

    objective = cp.Minimize(cp.trace(T_var))
    constraints = [T_var >> 0, cp.norm(g_var - G_vals, 1) <= ep]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CVXOPT)
    g_opt = g_var.value
    if g_var.value is None:
        raise ValueError("Optimization failed. Try increasing `ep` or checking input.")
    g_opt = g_opt / g_opt[0]
    return np.array(g_opt)


def compressed_sensing_extend(G_vals, ext_dim):
    len_t = len(G_vals)
    g_var = cp.Variable(ext_dim)

    T_entries = [[g_var[abs(i - j)] for j in range(ext_dim)] for i in range(ext_dim)]
    T_var = cp.bmat(T_entries)

    objective = cp.Minimize(cp.norm(g_var[0:len_t] - G_vals, 2))
    constraints = [T_var >> 0, cp.trace(T_var) == ext_dim]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CVXOPT)
    g_opt = g_var.value
    if g_var.value is None:
        raise ValueError("Optimization failed. Try increasing `ep` or checking input.")
    return np.array(g_opt)


def matrix_pencil_freq(y, dt, r=2, L=None):
    """Estimate angular frequencies from a sum of exponentials."""
    y = np.asarray(y, dtype=np.complex128).ravel()
    N = len(y)
    if L is None:
        L = N // 2
    if not (r < L < N - r + 1):
        raise ValueError("Need r < L < N-r+1. Try a larger N or different L.")

    Y0 = hankel(y[:L], y[L - 1:N - 1])
    Y1 = hankel(y[1:L + 1], y[L:N])

    U, s, Vh = svd(Y0, full_matrices=False)
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
    """Reconstruct a cosine-like signal after matrix-pencil frequency estimation."""
    A = np.column_stack([np.cos(omega * t), np.sin(omega * t)])
    coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    a, b = coeffs
    return a * np.cos(omega * t) + b * np.sin(omega * t)


####################
# MAIN ANALYSIS
####################

# Experimental parameters
n_values = [50]  # Number of qubits
N = 12  # Number of retained measurement points
k = 4  # Subsampling factor (every 4th point) of the full grid
alpha_idx = 50  # Total grid points in the full oscillation window
alpha_true_long = np.linspace(0, 2 * np.pi / n_values[0], alpha_idx, dtype=float)
alpha_true = alpha_true_long[0:k * N:k]  # Sampled points on the grid (12 in total)
R_trials = 25  # Number of Monte Carlo trials
imperfection_gamma = 0.1  # Additional experimental imperfection noise
# Hidden-dephasing sweep for the round-2 experiment.
# The estimators below do not receive the dephasing rate as input.
# Keep this moderate relative to the window length so the oscillation is still visible.
dephasing_rates = np.linspace(0.05*n_values[0]/(2*np.pi), n_values[0]/(2*np.pi), 7)
#dephasing_rates = [0,0,0,0]

ep = 0.25  # Base tolerance for compressed sensing

sample_idx = np.arange(0, k * N, k)
dt = (alpha_true_long[1] - alpha_true_long[0]) * k

err_shot = np.zeros((len(dephasing_rates), R_trials), dtype=float)
err_deco = np.zeros((len(dephasing_rates), R_trials), dtype=float)
err_denoise = np.zeros((len(dephasing_rates), R_trials), dtype=float)
err_mpen = np.zeros((len(dephasing_rates), R_trials), dtype=float)

nqubit = n_values[0]
for g_idx, dephasing_rate in enumerate(dephasing_rates):
    print(f"Processing dephasing_rate={dephasing_rate:.3f} ({g_idx + 1}/{len(dephasing_rates)})")
    for r in range(R_trials):
        clean_signal, shot_signal, deco_signal = generate_noisy_data(
            nqubit, alpha_true_long, imperfection_gamma, dephasing_rate=dephasing_rate
        )

        try:
            denoise_signal = compressed_sensing(deco_signal[sample_idx], ep + (0 * 0.01))
            err_denoise[g_idx, r] = fit_alpha(denoise_signal, nqubit, alpha_true) - nqubit
        except Exception:
            err_denoise[g_idx, r] = np.nan

        try:
            omega_hat, _, _ = estimate_single_cosine_frequency(deco_signal[sample_idx], dt, r=2)
            mpen_signal = recon_mpen_signal(alpha_true, deco_signal[sample_idx], omega_hat)
            err_mpen[g_idx, r] = fit_alpha(mpen_signal, nqubit, alpha_true) - nqubit
        except Exception:
            err_mpen[g_idx, r] = np.nan

        try:
            err_shot[g_idx, r] = fit_alpha(shot_signal[sample_idx], nqubit, alpha_true) - nqubit
        except Exception:
            err_shot[g_idx, r] = np.nan

        try:
            err_deco[g_idx, r] = fit_alpha(deco_signal[sample_idx], nqubit, alpha_true) - nqubit
        except Exception:
            err_deco[g_idx, r] = np.nan

alpha_shot = np.nanmean(err_shot, axis=1)
alpha_deco = np.nanmean(err_deco, axis=1)
alpha_denoise = np.nanmean(err_denoise, axis=1)
alpha_mpen = np.nanmean(err_mpen, axis=1)

yerr_shot = np.nanstd(err_shot, axis=1) / np.sqrt(R_trials)
yerr_deco = np.nanstd(err_deco, axis=1) / np.sqrt(R_trials)
yerr_cs = np.nanstd(err_denoise, axis=1) / np.sqrt(R_trials)
yerr_mpen = np.nanstd(err_mpen, axis=1) / np.sqrt(R_trials)



# Print relative estimation errors for the sweep
print("\nRelative estimation errors (averaged over trials; hidden dephasing):")
for idx, dr in enumerate(dephasing_rates):
    print(
        f"dephasing_rate={dr:.3f} | "
        f"Shot Noise: {abs(alpha_shot[idx]) / nqubit:.6f} | "
        f"Decoherence Noise: {abs(alpha_deco[idx]) / nqubit:.6f} | "
        f"Compressed Sensing: {abs(alpha_denoise[idx]) / nqubit:.6f} | "
        f"Matrix Pencil: {abs(alpha_mpen[idx]) / nqubit:.6f}"
    )

fig, ax = plt.subplots()

ax.errorbar(dephasing_rates* (2 * np.pi / nqubit), np.clip(np.abs(alpha_shot), 1e-12, None), yerr=yerr_shot,
            marker='o', color='0.7', label='Shot noise', capsize=3, lw=1)

ax.errorbar(dephasing_rates* (2 * np.pi / nqubit), np.clip(np.abs(alpha_deco), 1e-12, None), yerr=yerr_deco,
            marker='.', color='tab:red', label='Noisy data', capsize=3, lw=1)

ax.errorbar(dephasing_rates* (2 * np.pi / nqubit), np.clip(np.abs(alpha_mpen), 1e-12, None), yerr=yerr_mpen,
            marker='x', color='tab:purple', label='Matrix pencil', capsize=3, lw=1)

ax.errorbar(dephasing_rates* (2 * np.pi / nqubit), np.clip(np.abs(alpha_denoise), 1e-12, None), yerr=yerr_cs,
            marker='d', color='tab:green', label='CS (PSD)', capsize=3, lw=1)

ax.set_yscale('log')
ax.legend(fontsize=16)
ax.grid(True, which='both', linestyle='--', alpha=0.5)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
# plt.savefig("DephasingSweep.pdf")
plt.show()
