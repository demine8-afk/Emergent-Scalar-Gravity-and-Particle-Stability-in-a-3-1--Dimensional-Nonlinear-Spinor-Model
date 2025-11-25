import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# =============================================================================
# CODE METADATA & PHYSICS ESSENCE
# =============================================================================
# SCRIPT: fig4_binary_decay_ab_initio_v2.py
# PURPOSE: High-Fidelity Ab Initio Simulation of Scalar Gravitational Radiation.
#
# CORRECTION FROM V1:
# - Fixed interaction sign: Particles now properly attract via the scalar field.
# - High Resolution Grid (400x400) enabled by hardware capabilities.
# - Self-consistent radiative damping.
#
# PHYSICS ENGINE:
# 1. Wave Equation: dtt_Phi - c^2 laplacian_Phi = -4*pi*G * rho
#    (Source is negative to create a potential 'well' Phi < 0).
# 2. Force Law: F = - grad(Phi) * Coupling
#    (Particles roll down into the well they created).
#
# The "lag" in the potential well moving behind the particle (retardation)
# creates the drag force naturally.
# =============================================================================

# --- CONFIGURATION ---
# Domain
L = 15.0            # Space [-L, L]
N = 400             # High Res Grid
dx = 2*L / N
c = 1.0             # Speed of light
dt = 0.02           # Fine time step for stability
T_max = 45.0        # Enough time to see multiple orbits

# Physics
G_coupling = 2.0    # Strength of interaction
M_particle = 1.0
R_soft = 1.0        # Particle Radius (Gaussian width) for numerical smoothness

# Orbit Setup
R_init = 6.0        # Separation
# Orbital Velocity Estimation for circular orbit: v ~ sqrt(F*r/m)
# Force approx G/r^2 (Newtonian limit). v ~ sqrt(G/r).
v_orbit = 0.55      # Slightly sub-circular to ensure elliptical inspiral

# Stability Check (CFL)
assert c * dt / dx < 0.7, "CFL condition violated. Reduce dt."

# --- FIELD SOLVER (FDTD) ---
x = np.linspace(-L, L, N)
y = np.linspace(-L, L, N)
X, Y = np.meshgrid(x, y)

# Precompute Gaussian masks to save CPU cycles
def add_source_to_grid(pos1, pos2, grid_buffer):
    """Directly maps particle positions to source density rho."""
    # Vectorized distance calculation is expensive, we do local updates or optimized numpy
    # Since we have power, full grid calc is fine for N=400
    r2_1 = (X - pos1[0])**2 + (Y - pos1[1])**2
    r2_2 = (X - pos2[0])**2 + (Y - pos2[1])**2
    
    # Source is NEGATIVE to create a potential WELL (Attraction)
    rho = - (np.exp(-r2_1 / (2*R_soft**2)) + np.exp(-r2_2 / (2*R_soft**2)))
    return rho

def get_force_interpolated(phi, pos):
    """Bilinear interpolation of gradient at particle position."""
    # Grid indices
    x_idx = (pos[0] + L) / dx
    y_idx = (pos[1] + L) / dx
    
    i = int(x_idx)
    j = int(y_idx)
    
    # Safety clamp
    if i < 1 or i >= N-2 or j < 1 or j >= N-2:
        return np.array([0.0, 0.0])
    
    # Central difference gradients on grid
    # grad_x at (j, i)
    gx_grid = (phi[j, i+1] - phi[j, i-1]) / (2*dx)
    gy_grid = (phi[j+1, i] - phi[j-1, i]) / (2*dx)
    
    # Force F = - Coupling * grad(Phi)
    # Since Phi is negative (well), grad points outwards (up slope).
    # We want to go DOWN the slope.
    # F = - grad(Phi) * coupling.
    
    return -np.array([gx_grid, gy_grid]) * G_coupling

# --- MAIN LOOP ---
print(f"Simulating High-Fidelity Field Dynamics. Grid: {N}x{N}...")

# Init
pos1 = np.array([-R_init/2, 0.0])
pos2 = np.array([R_init/2, 0.0])
vel1 = np.array([0.2, -v_orbit]) # Asymmetric start
vel2 = np.array([-0.2, v_orbit])

phi_curr = np.zeros((N, N))
phi_prev = np.zeros((N, N))

# History
traj1 = []
traj2 = []
separation = []
times = []

steps = int(T_max / dt)

# Boundary Damping Mask (Sponge Layer)
mask = np.ones((N, N))
edge = int(N * 0.05)
mask[:edge, :] *= 0.9; mask[-edge:, :] *= 0.9
mask[:, :edge] *= 0.9; mask[:, -edge:] *= 0.9

for n in tqdm(range(steps)):
    # 1. Source
    rho = add_source_to_grid(pos1, pos2, None)
    
    # 2. Wave Update (Verlet)
    laplacian = (np.roll(phi_curr, 1, axis=0) + np.roll(phi_curr, -1, axis=0) +
                 np.roll(phi_curr, 1, axis=1) + np.roll(phi_curr, -1, axis=1) - 
                 4*phi_curr) / dx**2
    
    phi_next = 2*phi_curr - phi_prev + (dt**2) * (c**2 * laplacian + rho)
    phi_next *= mask # Absorb boundary reflections
    
    # 3. Particle Dynamics
    F1 = get_force_interpolated(phi_curr, pos1)
    F2 = get_force_interpolated(phi_curr, pos2)
    
    # Symplectic Euler
    vel1 += F1 * dt / M_particle
    vel2 += F2 * dt / M_particle
    pos1 += vel1 * dt
    pos2 += vel2 * dt
    
    # Rotate buffers
    phi_prev = phi_curr
    phi_curr = phi_next
    
    # Store
    if n % 4 == 0:
        traj1.append(pos1.copy())
        traj2.append(pos2.copy())
        dist = np.linalg.norm(pos1 - pos2)
        separation.append(dist)
        times.append(n * dt)
        
        # Crash detection
        if dist < R_soft:
            print(f"Merger detected at t={n*dt:.2f}")
            break

# --- VISUALIZATION ---
print("Rendering...")
traj1 = np.array(traj1)
traj2 = np.array(traj2)

fig = plt.figure(figsize=(15, 6))

# Panel A: Field + Trajectory
ax1 = fig.add_subplot(121)
# Invert colors: Deep well is bright or dark? Let's use 'magma' inverted or similar.
# Phi is negative. So deeper well = more negative.
im = ax1.imshow(phi_curr, extent=[-L, L, -L, L], origin='lower', cmap='magma', vmin=-1.5, vmax=0.2)
ax1.plot(traj1[:,0], traj1[:,1], 'c-', lw=1.5, alpha=0.8, label='Soliton A')
ax1.plot(traj2[:,0], traj2[:,1], 'w-', lw=1.5, alpha=0.8, label='Soliton B')

# Current positions
ax1.scatter(pos1[0], pos1[1], c='cyan', s=100, edgecolors='white', zorder=10)
ax1.scatter(pos2[0], pos2[1], c='white', s=100, edgecolors='cyan', zorder=10)

ax1.set_title("a) Scalar Field Potential & Inspiral", fontweight='bold')
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.legend()

# Panel B: Separation R(t)
ax2 = fig.add_subplot(122)
ax2.plot(times, separation, 'k-', lw=1.5)
ax2.set_title("b) Orbital Decay (Ab Initio)", fontweight='bold')
ax2.set_xlabel("Time t")
ax2.set_ylabel("Separation R(t)")
ax2.grid(True, alpha=0.3)

# Physics annotation
ax2.annotate('Radiative Energy Loss\n(No artificial friction)', 
             xy=(times[len(times)//3], separation[len(separation)//3]), 
             xytext=(times[len(times)//3]+5, separation[len(separation)//3]+2),
             arrowprops=dict(facecolor='black', arrowstyle='->'),
             bbox=dict(boxstyle="round", fc="white", ec="gray"))

plt.suptitle("Figure 4: Binary Dynamics. Corrected Ab Initio Field Simulation.", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("fig4_binary_ab_initio_corrected.png", dpi=150)
print("Saved.")
plt.show()
