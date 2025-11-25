import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import time

# =============================================================================
# CODE METADATA & PHYSICS ESSENCE 
# =============================================================================
# SCRIPT: Coherent_scalar_interaction.py
# PURPOSE: Demonstration of Wave-Like Gravitational Interaction (Interference).
#
# THEORETICAL BACKGROUND:
# In Scalar Field Dark Matter (SFDM) models, the interaction between two 
# bosonic cores is phase-dependent. Unlike classical particles that always attract,
# coherent scalar waves can exhibit repulsive behavior due to interference.
#
# 1. Constructive Interference (Delta_phi = 0):
#    The density between cores increases, deepening the potential well.
#    Result: Enhanced attraction and merger.
#
# 2. Destructive Interference (Delta_phi = pi):
#    A nodal plane (Psi = 0) forms between the cores. By Heisenberg principle,
#    squeezing the wavefunction against this node creates infinite gradient pressure.
#    Result: "Quantum Pressure" repulsion that overpowers gravity.
#
# NUMERICAL METHOD:
# Full 3D Split-Step Fourier spectral evolution of the Schrödinger-Poisson system
# on a 128^3 grid.
# =============================================================================

# =============================================================================
# CONFIGURATION
# =============================================================================
N = 128
L = 10.0          # Domain size [kpc]
dt = 0.01         # Time step
steps = 600       # Total simulation steps
store_every = 20  # Visualization sampling rate

# Spatial Grid
dz = L / N
x = np.linspace(-L/2, L/2, N)
X, Y, Z = np.meshgrid(x, x, x, indexing='ij')

# Spectral (k-space) Grid
k = np.fft.fftfreq(N, d=dz) * 2 * np.pi
KX, KY, KZ = np.meshgrid(k, k, k, indexing='ij')
K_sq = KX**2 + KY**2 + KZ**2
K_sq[0, 0, 0] = 1.0 # Avoid division by zero (mean field correction)

# Pre-computed Evolution Operator (Kinetic Term)
Op_K_half = np.exp(-1j * K_sq * (dt / 2))

print("="*50)
print("3D SFDM PHASE-DEPENDENT INTERACTION (High-Res)")
print(f"Grid: {N}³ = {N**3:,} points")
print("="*50)

# =============================================================================
# SIMULATION KERNEL
# =============================================================================

def run_simulation(phase_diff, label=""):
    """
    Evolves two Gaussian solitons with a specific phase difference.
    """
    print(f"\nRunning: {label}")
    
    sigma = 0.8
    sep_init = 3.0
    
    # Initial Wave Packets construction
    R1_sq = (X + sep_init/2)**2 + Y**2 + Z**2
    R2_sq = (X - sep_init/2)**2 + Y**2 + Z**2
    
    psi1 = np.exp(-R1_sq / (2*sigma**2), dtype=complex)
    psi2 = np.exp(-R2_sq / (2*sigma**2), dtype=complex)
    psi2 *= np.exp(1j * phase_diff) # Apply phase rotation
    
    Psi = psi1 + psi2
    
    # Normalization (Mass scaling)
    norm_factor = np.sqrt(np.sum(np.abs(Psi)**2) * dz**3)
    Psi /= norm_factor
    Psi *= 3.0 # Effective mass parameter for observable interaction
    
    separations = []
    slices = []
    mid = N // 2 # Middle index for Z-slice
    
    for step in range(steps):
        # --- Step 1: Drift (Kinetic / half-step) ---
        Psi = np.fft.ifftn(np.fft.fftn(Psi) * Op_K_half)
        
        # --- Step 2: Kick (Potential / full-step) ---
        rho = np.abs(Psi)**2
        
        # Solve Poisson Eq: -k^2 * Phi_k = -4*pi*rho_k
        Phi_k = -4 * np.pi * np.fft.fftn(rho) / K_sq
        Phi_k[0, 0, 0] = 0.0 # Zero mode removal
        Phi = np.real(np.fft.ifftn(Phi_k))
        
        # Apply Phase kick
        Psi *= np.exp(-1j * Phi * dt)
        
        # --- Step 3: Drift (Kinetic / half-step) ---
        Psi = np.fft.ifftn(np.fft.fftn(Psi) * Op_K_half)
        
        # --- Diagnostics ---
        if step % store_every == 0:
            # Measure Separation:
            # Project 3D density to 1D X-axis
            rho_x = np.sum(rho, axis=(1, 2))
            rho_smooth = gaussian_filter1d(rho_x, sigma=5)
            
            # Find peaks
            idx_L = np.argmax(rho_smooth[:mid])
            idx_R = np.argmax(rho_smooth[mid:]) + mid
            sep = abs(x[idx_R] - x[idx_L])
            
            separations.append(sep)
            slices.append(rho[:, :, mid].copy()) # Store central plane slice
            
            if step % 100 == 0:
                print(f"  Step {step}/{steps}, Separation = {sep:.2f} kpc")
    
    return np.array(separations), np.array(slices)

# =============================================================================
# EXECUTION
# =============================================================================
t_start = time.time()

# Simulation 1: Constructive Interference
sep1, slices1 = run_simulation(0.0, "In-Phase (Δφ=0)")

# Simulation 2: Destructive Interference
sep2, slices2 = run_simulation(np.pi, "Anti-Phase (Δφ=π)")

print(f"\n✓ Total calculation time: {time.time() - t_start:.1f} sec")

# Smoothing for publication-quality curves
sep1_smooth = gaussian_filter1d(sep1, sigma=2)
sep2_smooth = gaussian_filter1d(sep2, sigma=2)

print(f"\nIn-phase final dist:   {sep1_smooth[-1]:.2f} kpc (Merger)")
print(f"Anti-phase final dist: {sep2_smooth[-1]:.2f} kpc (Repulsion)")

# =============================================================================
# VISUALIZATION
# =============================================================================
times = np.arange(len(sep1)) * store_every * dt

fig = plt.figure(figsize=(14, 10), facecolor='white')

# --- Panel A: Separation Dynamics ---
ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(times, sep1_smooth, 'b-', lw=3.0, label=r'In-phase ($\Delta\phi=0$)')
ax1.plot(times, sep2_smooth, 'r--', lw=3.0, label=r'Anti-phase ($\Delta\phi=\pi$)')

ax1.set_xlabel('Time [code units]', fontsize=12)
ax1.set_ylabel('Separation [kpc]', fontsize=12)
ax1.set_title('(a) 3D Soliton Separation Dynamics', fontweight='bold', fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 8.0) 

# Common Plotting Parameters
extent = [-L/2, L/2, -L/2, L/2]
vmax = np.max(slices1[0]) * 0.8 
cmap = 'inferno'

# --- Panel B: Initial ---
ax2 = fig.add_subplot(2, 2, 2)
im2 = ax2.imshow(slices1[0].T, origin='lower', extent=extent, cmap=cmap, vmax=vmax)
ax2.set_title('(b) Initial State (t=0)', fontweight='bold', fontsize=14)
ax2.set_xlabel('x [kpc]')
ax2.set_ylabel('y [kpc]')

# --- Panel C: Merged (In-Phase) ---
ax3 = fig.add_subplot(2, 2, 3)
im3 = ax3.imshow(slices1[-1].T, origin='lower', extent=extent, cmap=cmap, vmax=vmax)
ax3.set_title('(c) In-Phase: GRAVITATIONAL COLLAPSE', fontweight='bold', color='blue', fontsize=14)
ax3.set_xlabel('x [kpc]')
ax3.set_ylabel('y [kpc]')
ax3.text(0, 0, 'MERGER', color='white', fontsize=12, fontweight='bold', ha='center', va='center',
         bbox=dict(facecolor='blue', alpha=0.3, edgecolor='none'))

# --- Panel D: Separated (Anti-Phase) ---
ax4 = fig.add_subplot(2, 2, 4)
im4 = ax4.imshow(slices2[-1].T, origin='lower', extent=extent, cmap=cmap, vmax=vmax)
ax4.set_title('(d) Anti-Phase: QUANTUM REPULSION', fontweight='bold', color='red', fontsize=14)
ax4.set_xlabel('x [kpc]')
ax4.set_ylabel('y [kpc]')

# Highlight the Nodal Plane
ax4.axvline(0, color='cyan', ls='--', lw=2, alpha=0.8)
ax4.text(0, 3.5, 'NODAL PLANE\n(DARK SOLITON)', color='cyan', fontsize=9, fontweight='bold',
         ha='center', bbox=dict(facecolor='black', alpha=0.6, edgecolor='none'))

plt.suptitle('Figure 12: Interference-Modulated Gravitational Interaction in SFDM', 
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()

# Save
filename = 'Coherent_scalar_interaction.png'
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"\nFigure saved as '{filename}'")
plt.show()
