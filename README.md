# Emergent Scalar Gravity and Quantum Analogues in a (3+1)-Dimensional Nonlinear Spinor Model

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17699433.svg)](https://doi.org/10.5281/zenodo.17699433)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Abstract
This repository contains the numerical source code and validation scripts for the preprint **"Emergent Scalar Gravity and Quantum Analogues in a (3+1)-Dimensional Nonlinear Spinor Model"**.

We investigate a field-theoretic framework where "particles" arise as topological solitons (breathers) of a nonlinear Dirac field, stabilized by a Soler-type self-interaction. These extended spinor fields are coupled to a massless scalar field, which mediates an attractive long-range force mimicking gravity. The simulations demonstrate that this classical field theory self-consistently reproduces:
1.  **Quantum phenomena:** Wave-particle duality, tunneling, and interference.
2.  **Relativistic gravity:** Orbital decay via radiation, gravitational lensing, and redshift.
3.  **Astrophysics:** Galactic rotation curves without Dark Matter (via scalar halos).

---

## Theoretical Framework

The dynamics are governed by the action $S = \int d^4x \sqrt{-g}\mathcal{L}$ in flat spacetime with the following Lagrangian density:

$$
\mathcal{L} = \bar{\Psi}(i\slashed{\partial} - m)\Psi + \mathcal{L}_{\text{NL}} + \mathcal{L}_{\text{grav}}
$$

### 1. Nonlinear Stabilization (Soler Model)
To prevent dispersive decay, the spinor field possesses a saturating nonlinearity:
$$
\mathcal{L}_{\text{NL}} = \int_0^{\bar{\Psi}\Psi} V(\rho') d\rho', \quad V(\rho) = -G_S \frac{\rho}{1 + S \rho}
$$

### 2. Scalar Gravity Coupling
The matter sector couples to a massless scalar field $\Phi$, satisfying the relativistic Poisson equation:
$$
\Box \Phi = -4\pi G \xi \bar{\Psi}\Psi
$$
This creates an effective position-dependent mass or, equivalently, a curved metric background:
$$
g_{\mu\nu}^{\text{eff}} \approx \eta_{\mu\nu}(1 + 2\Phi)
$$

---

## Numerical Implementation & Reproducibility

The code implements high-precision split-step Fourier methods (SSFM) and finite-difference schemes to solve the coupled Dirac-Klein-Gordon system. All scripts are standalone and generate the figures presented in the paper.

### **Core Dynamics & Stability**
| Script File | Description | Physics Checked |
| :--- | :--- | :--- |
| `fig1_stability_soliton_3d.py` | **Soliton Stability** | Verifies that the nonlinearity $V(\rho)$ stabilizes the wave packet against dispersion. |
| `fig2_unitarity_conservation_test.py` | **Conservation Laws** | Checks global Noether charge conservation $Q = \int \Psi^\dagger \Psi d^3x$ (Unitarity). |
| `fig3_relativistic_scattering_soler.py` | **Scattering** | Simulates relativistic head-on collisions to demonstrate topological robustness (elasticity). |

### **Emergent Gravity & Relativity**
| Script File | Description | Physics Checked |
| :--- | :--- | :--- |
| `fig4_binary_decay_final.py` | **Binary Decay** | Orbital inspiral of a binary system due to scalar radiation (energy loss). |
| `fig5_lorentz_contraction_wakefield.py` | **Lorentz Contraction** | Visualization of potential contraction for a source moving at $v \approx c$. |
| `fig6_gravitational_lensing.py` | **Lensing** | Refraction of wave packets passing through a scalar potential well ($\Phi < 0$). |
| `fig7_gravitational_redshift.py` | **Redshift** | Spectral shift of eigenfrequencies for a soliton inside a gravitational potential. |

### **Quantum Analogues**
| Script File | Description | Physics Checked |
| :--- | :--- | :--- |
| `fig8_quantum_interference.py` | **Interference** | Double-slit experiment simulation showing fringe patterns from classical fields. |
| `fig9_tunneling_effect.py` | **Tunneling** | Barrier penetration probability where $E_{kin} < V_{barrier}$. |

### **Astrophysics**
| Script File | Description | Physics Checked |
| :--- | :--- | :--- |
| `fig11_galactic_rotation.py` | **Rotation Curves** | Least-Squares fit of NGC 6503, NGC 2841, and DDO 154 using the Scalar Halo model ($\chi^2_\nu < 1$). |

---

## Usage

### Dependencies
The simulations require standard scientific Python packages:

    pip install numpy matplotlib scipy

### Running a Simulation
To reproduce any figure (e.g., the Tunneling effect), simply run the corresponding script:

    python fig9_tunneling_effect.py

The script will perform the physics calculation and save the output plot as a `.png` file in the working directory.

---

## Citation

If you use this code or framework in your research, please cite the dataset/codebase via Zenodo:

    Danilov, S. A. (2025). Emergent Scalar Gravity and Particle Stability in a (3+1)-Dimensional Nonlinear Spinor Model [Data set]. Zenodo. https://doi.org/10.5281/zenodo.17699433

---
*For theoretical inquiries or collaboration, please refer to the contact information provided in the preprint.*
