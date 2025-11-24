# Numerical Stress-Tests & Validation

This gallery contains additional simulation results performed to validate the robustness of the (3+1)D nonlinear spinor model beyond the scope of the main preprint.

The code implements a Split-Step Fourier Method (SSFM) solver for the coupled Dirac-Scalar system.

### 1. General Relativity Test: Gravitational Lensing
Demonstration of light ray deflection by a massive scalar potential center. The model correctly reproduces the bending of relativistic trajectories ($\beta \approx 0.95$).
*(See `test_gravitational_lensing.png`)*

### 2. Quantum Mechanics: Double-Slit Interference
Wave-particle duality test. A soliton passes through a double-slit barrier, exhibiting a classic diffraction pattern and self-interference on the detector screen.
*(See `test_double_slit.png`)*

### 3. Spin Dynamics: Larmor Precession
Unitary evolution of the internal spinor degrees of freedom in an external magnetic field. $\langle S_x \rangle$ and $\langle S_y \rangle$ oscillate while preserving the norm, confirming the validity of the operator splitting scheme.
*(See `test_spin_precession.png`)*

### 4. Orbital Mechanics: Binary Capture
Dynamical capture of two massive solitons into a bound orbit solely via mutual scalar attraction. Proves the emergence of long-range forces from local field interactions.
*(See `test_binary_orbit.png`)*
