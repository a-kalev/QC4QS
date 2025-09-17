# Quantum Sensing with Compressed Sensing Denoising

A Python implementation of compressed sensing techniques for denoising quantum sensing measurements, exploiting the Toeplitz structure of quantum correlation functions.

## Overview

This code demonstrates how compressed sensing can improve parameter estimation in quantum sensing experiments by denoising measurements corrupted by shot noise and decoherence effects.

## Key Features

- **Quantum Signal Generation**: Simulates ideal quantum signals with realistic noise models
- **Compressed Sensing Denoising**: Uses semidefinite programming to exploit Toeplitz structure
- **Parameter Estimation**: Fits qubit numbers from noisy measurements using curve fitting
- **Performance Analysis**: Compares estimation accuracy across different noise conditions

## Requirements

```bash
numpy
matplotlib
scipy
cvxpy
```

## Installation

```bash
pip install numpy matplotlib scipy cvxpy
```

## Usage

Can be run directly on Jupiter

The script will:
1. Generate noisy quantum sensing data
2. Apply compressed sensing denoising
3. Compare parameter estimation performance
4. Display results and save visualization

## Key Functions

- `generate_noisy_data()`: Creates quantum signals with shot noise and decoherence
- `compressed_sensing_denoise()`: Applies Toeplitz-based denoising
- `fit_qubit_number()`: Estimates parameters from measurements

## Output

- Console output showing relative estimation errors
- Visualization comparing ideal, noisy, and denoised signals
- SVG plot saved as `quantum_sensing_results.svg`

## Physics

The method exploits the fact that quantum correlation functions have a Toeplitz structure, enabling efficient denoising through trace minimization under positive semidefinite constraints.

## Citation

Please cite as github repo: akalev/CS4QS 
