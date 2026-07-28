#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 2026

@author: amirk
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz, hankel, svd, eig
import cvxpy as cp
from scipy.optimize import curve_fit


def generate_noisy_data(amplitudes, frequencies, gamma):
    ideal = np.cos(amplitudes * frequencies)
    sigma_shot = (ideal**2) * (np.full(len(frequencies), 1.0) - (ideal**2)) / 50
    deco = np.cos(frequencies) ** amplitudes
    sigma_deco = (deco**2) * (np.full(len(frequencies), 1.0) - (deco**2)) / 50

    shot_gnal = np.random.normal(ideal, sigma_shot)
    deco_gnal = (1 - gamma) * shot_gnal + gamma * np.random.normal(0, 0.25, len(frequencies))  # gamma*np.random.normal(deco,sigma_deco)

    return ideal, shot_gnal, deco_gnal


def fit_alpha(G_vals, amplitudes, frequencies):
    def model(frequencies, nfit):
        return np.cos(nfit * frequencies)

    popt, pcov = curve_fit(model, frequencies, G_vals, p0=[0.8 * amplitudes])
    qubit_est = popt[0]
    return qubit_est


def compressed_sensing(G_vals, ep):
    # Setup the optimization
    len_t = len(G_vals)
    g_var = cp.Variable(len_t)

    # Define symmetric Toeplitz matrix from cos_row
    T_entries = [[g_var[abs(i - j)] for j in range(len_t)] for i in range(len_t)]
    T_var = cp.bmat(T_entries)

    # Objective: minimize Trace norm to original T
    objective = cp.Minimize(cp.trace(T_var))
    # Constraints:
    constraints = [T_var >> 0, cp.norm(g_var - G_vals, 1) <= ep]
    # Solve the problem
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CVXOPT)
    g_opt = g_var.value
    if g_var.value is None:
        raise ValueError("Optimization failed. Try increasing `ep` or checking input.")
    g_opt = g_opt / g_opt[0]
    return np.array(g_opt)


def compressed_sensing_extend(G_vals, ext_dim):
    # Setup the optimization
    len_t = len(G_vals)
    g_var = cp.Variable(ext_dim)

    # Define symmetric Toeplitz matrix from cos_row
    T_entries = [[g_var[abs(i - j)] for j in range(ext_dim)] for i in range(ext_dim)]
    T_var = cp.bmat(T_entries)

    # Objective: minimize Frobenius norm to original T
    objective = cp.Minimize(cp.norm(g_var[0:len_t] - G_vals, 2))
    # Constraints:
    constraints = [T_var >> 0, cp.trace(T_var) == ext_dim]
    # Solve the problem
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CVXOPT)
    g_opt = g_var.value
    #g_opt=g_opt/g_opt[0]
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
    # Fit y ≈ a*cos(ωt) + b*sin(ωt)
    A = np.column_stack([np.cos(omega*t), np.sin(omega*t)])
    coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    a, b = coeffs
    return a*np.cos(omega*t) + b*np.sin(omega*t)


####################
# MAIN ANALYSIS
####################

# Experimental parameters
n_values = [50]          # Number of qubits
K_list = [8, 10, 12, 14, 16]   # Number of retained samples
alpha_idx = 50           # Total grid points
R_trials = 25
gammas = 0.1


alpha_true_long = np.linspace(0, 2 * np.pi / n_values[0], alpha_idx, dtype=float)

err_shot = np.zeros((len(K_list), R_trials), dtype=float)
err_deco = np.zeros((len(K_list), R_trials), dtype=float)
err_denoise = np.zeros((len(K_list), R_trials), dtype=float)
err_mpen = np.zeros((len(K_list), R_trials), dtype=float)

for k_idx, K in enumerate(K_list):
    print(f"Processing K={K} ({k_idx + 1}/{len(K_list)})")

    sample_idx = np.linspace(0, alpha_idx - 1, K, dtype=int)
    alpha_true = alpha_true_long[sample_idx]
    dt = alpha_true[1] - alpha_true[0]
    ep = 0.025*K
    for r in range(R_trials):
        clean_signal, shot_signal, deco_signal = generate_noisy_data(n_values[0], alpha_true_long, gammas)

        shot_samp = shot_signal[sample_idx]
        deco_samp = deco_signal[sample_idx]

        try:
            denoise_signal = compressed_sensing(deco_samp, ep)
            err_denoise[k_idx, r] = fit_alpha(denoise_signal, n_values[0], alpha_true) - n_values[0]
        except ValueError:
            err_denoise[k_idx, r] = np.nan

        try:
            omega_hat, _, _ = estimate_single_cosine_frequency(deco_samp, dt, r=2)
            mpen_signal = recon_mpen_signal(alpha_true, deco_samp, omega_hat)
            err_mpen[k_idx, r] = fit_alpha(mpen_signal, n_values[0], alpha_true) - n_values[0]
        except ValueError:
            err_mpen[k_idx, r] = np.nan

        err_shot[k_idx, r] = fit_alpha(shot_samp, n_values[0], alpha_true) - n_values[0]
        err_deco[k_idx, r] = fit_alpha(deco_samp, n_values[0], alpha_true) - n_values[0]
        
alpha_shot = np.nanmean(err_shot, axis=1)
alpha_deco = np.nanmean(err_deco, axis=1)
alpha_denoise = np.nanmean(err_denoise, axis=1)
alpha_mpen = np.nanmean(err_mpen, axis=1)

# Print relative estimation errors
print(f"\nRelative estimation errors:")
print(f"Shot Noise: {abs(alpha_shot) / n_values[0]}")
print(f"Decoherence Noise: {abs(alpha_deco) / n_values[0]}")
print(f"Compressed Sensing: {abs(alpha_denoise) / n_values[0]}")
print(f"Matrix Pencil: {abs(alpha_mpen) / n_values[0]}")


fig, ax = plt.subplots()

K_vals = np.array(K_list)

y_shot = np.abs(alpha_shot) / n_values[0]
y_deco = np.abs(alpha_deco) / n_values[0]
y_cs   = np.abs(alpha_denoise) / n_values[0]
y_mpen = np.abs(alpha_mpen) / n_values[0]

ax.plot(K_vals, y_shot,  marker='o', color='0.7', label='Shot Noise')
ax.plot(K_vals, y_deco,  marker='.', color='tab:red', label='Measured')
ax.plot(K_vals, y_mpen,  marker='x', color='tab:purple', label='Matrix pencil')
ax.plot(K_vals, y_cs,    marker='d', color='tab:green', label='This work')

ax.set_yscale('log')

ax.set_xlabel('Number of samples $K$', fontsize=16)
ax.set_ylabel(r'Relative error $\delta\alpha$', fontsize=16)

ax.legend(fontsize=12)
ax.grid(True, which='both', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()