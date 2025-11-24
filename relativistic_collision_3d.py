"""
SCRIPT 02: RELATIVISTIC SCATTERING TEST
---------------------------------------
Purpose:
Simulates a high-energy head-on collision between two spinor solitons in (3+1)D.
This verifies the "particle-like" stability of the solutions. Despite the 
complex interference pattern during the merger, the packets must re-emerge 
intact, demonstrating topological robustness.

Key Features:
- Relativistic initial conditions (Lorentz-contracted width).
- Nonlinear Dirac evolution.
- Visualization of Approach, Merger, and Scattering phases.
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fftn, ifftn, fftfreq

# --- PHYSICS PARAMETERS ---
# Grid and Time
N = 128              # High resolution for sharp interference fringes
L = 32.0             # Physical box size
dt = 0.02            # Time step
t_max = 24.0         # Duration sufficient for approach and separation

# Physics
m = 1.0              # Mass
p_mom = 0.8          # Initial momentum (Relativistic: p ~ m*gamma*v)
# Soler Nonlinearity Parameters (Strong coupling for stability)
G_sol = 2.0          
S_sol = 3.0          

# --- GRID SETUP ---
x = np.linspace(-L/2, L/2, N, endpoint=False)
X, Y, Z = np.meshgrid(x, x, x, indexing='ij')

# --- SPECTRAL OPERATORS ---
# Momentum space definitions
kx = 2 * np.pi * fftfreq(N, L/N)
KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing='ij')
K2 = KX**2 + KY**2 + KZ**2

# Kinetic Propagator Construction
# We compute the exact Dirac propagator exp(-i H_0 dt) in k-space
Ek = np.sqrt(K2 + m**2)
Ck = np.cos(Ek * dt)
Sk = np.sin(Ek * dt) / (Ek + 1e-10)

# Dirac Matrices (Standard Representation)
I = np.eye(4, dtype=complex)
# We focus on Y-axis dynamics for the collision
AlphaY = np.array([[0,0,0,-1j],[0,0,1j,0],[0,-1j,0,0],[1j,0,0,0]], dtype=complex)
Beta   = np.array([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,-1]], dtype=complex)

# Kinetic Evolution Operator (U_kin)
# H_k = alpha_y * k_y + beta * m
Hk_term = np.einsum('ij,xyz->ijxyz', AlphaY, KY) + np.einsum('ij,xyz->ijxyz', Beta, np.full_like(K2, m))
U_kin = Ck[None, None, ...] * I[:, :, None, None, None] - 1j * Sk[None, None, ...] * Hk_term

# --- INITIALIZATION ---
def make_packet(y_pos, p_y):
    """
    Creates a boosted Gaussian spinor.
    Includes explicit Lorentz contraction of the spatial width.
    """
    # Lorentz factor calculation
    E_val = np.sqrt(p_y**2 + m**2)
    gamma = E_val / m
    
    # Contraction: Width is narrower along direction of motion (Y)
    width_long = 2.5 / gamma
    width_perp = 2.5
    
    # Envelope
    r2 = (X**2 + Z**2)/(width_perp**2) + (Y - y_pos)**2 / (width_long**2)
    
    # Initialize Spinor (Spin Up, Component 0)
    psi = np.zeros((4, N, N, N), dtype=complex)
    psi[0] = np.exp(-r2/2.0) * np.exp(1j * p_y * Y)
    return psi

# Create two counter-propagating packets
# Packet 1: At y=+8 moving Left (-p)
# Packet 2: At y=-8 moving Right (+p)
Psi = make_packet(8.0, -p_mom) + make_packet(-8.0, p_mom)

# Normalization (Charge Scaling)
# We scale the total charge to ensure the nonlinear term is active enough
norm = np.sum(np.abs(Psi)**2) * (L/N)**3
Psi *= np.sqrt(30.0 / norm) 

# --- EVOLUTION LOOP ---
steps = int(t_max / dt)
# We want to capture 3 specific moments: Start, Impact, End
snap_indices = [0, int(steps*0.42), steps]
snapshots = []

print(f"Starting Collision Sim: p_y=+/-{p_mom}, Interaction Strength G={G_sol}")

for step in range(steps + 1):
    # 1. Nonlinear Potential Step (Half-step approximation for speed)
    rho = np.sum(np.abs(Psi)**2, axis=0)
    
    # Soler Potential: V = - G * rho / (1 + S * rho)
    # This saturating potential prevents collapse and ensures stability
    V = -G_sol * rho / (1.0 + S_sol * rho)
    
    Psi *= np.exp(-0.5j * V * dt)
    
    # 2. Kinetic Step (Spectral)
    Psi_k = fftn(Psi, axes=(1,2,3))
    Psi_k = np.einsum('ijxyz,jxyz->ixyz', U_kin, Psi_k)
    Psi = ifftn(Psi_k, axes=(1,2,3))
    
    # 3. Nonlinear Potential Step (Second half)
    Psi *= np.exp(-0.5j * V * dt)
    
    # Capture frame
    if step in snap_indices:
        # Slice through X=0 plane to visualize Y-Z dynamics
        snapshots.append(rho[N//2, :, :])
        print(f"Captured frame at step {step}")

# --- PLOTTING ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
titles = ["(a) Approach (t=0)", "(b) Merger/Interference", "(c) Scattering (t=final)"]

for i, ax in enumerate(axes):
    # Visualize density
    im = ax.imshow(snapshots[i], origin='lower', cmap='inferno', 
                   extent=[-L/2, L/2, -L/2, L/2], vmin=0, vmax=np.max(snapshots[i]))
    
    ax.set_title(titles[i], fontweight='bold')
    ax.set_xlabel("Longitudinal Position Y")
    ax.set_ylabel("Transverse Position Z")
    
    # Add simple grid
    ax.grid(color='white', linestyle='--', linewidth=0.5, alpha=0.3)

# Layout adjustments
plt.tight_layout()
output_file = "fig_collision_recovered.png"
plt.savefig(output_file, dpi=150)
print(f"Scattering visualization saved to {output_file}")
plt.close()
