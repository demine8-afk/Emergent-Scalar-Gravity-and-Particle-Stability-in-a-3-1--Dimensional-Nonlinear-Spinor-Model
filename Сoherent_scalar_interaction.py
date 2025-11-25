import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# =============================================================================
# CODE METADATA & PHYSICS ESSENCE
# =============================================================================
# SCRIPT: interference_modulated_gravity.py
# PURPOSE: Analysis of Phase-Dependent Interaction in Coherent Scalar Fields.
#
# THEORETICAL CONTEXT:
# Unlike dust or point particles in GR where rho_tot = rho1 + rho2, 
# coherent scalar fields obey superposition of amplitudes: 
# rho_tot = |Psi1 + Psi2|^2 = rho1 + rho2 + 2*sqrt(rho1*rho2)*cos(dPhi).
#
# DOMAIN OF VALIDITY:
# This effect is negligible for incoherent matter (thermal gas, planets) where 
# phase averages to zero. However, it is dominant for:
# 1. Macroscopic Quantum States (BECs, Superfluids).
# 2. Scalar Field Dark Matter (SFDM / Fuzzy Dark Matter) cores.
# 3. Boson Star mergers.
#
# SIMULATION:
# We simulate the interaction of two "breather" solitons with controlled phase difference.
# =============================================================================

# --- CONFIGURATION ---
L = 40.0
N = 1024
dt = 0.05
t_max = 60.0

# Physics Parameters
mass = 1.0
G_soler = 1.5
S_soler = 0.5
xi = 0.8        # Scalar coupling
G_newton = 5.0  # Effective gravitational coupling

# Grid
z = np.linspace(-L/2, L/2, N)
dz = z[1] - z[0]

# --- SOLVER KERNELS ---

def solve_poisson(rho):
    """ Solves d2Phi/dz2 = 4*pi*G*rho in Fourier space """
    k = np.fft.fftfreq(N, d=dz) * 2 * np.pi
    rho_k = np.fft.fft(rho)
    k[0] = 1.0
    Phi_k = (4 * np.pi * G_newton * xi * rho_k) / (k**2)
    Phi_k[0] = 0.0
    return np.real(np.fft.ifft(Phi_k))

def get_separation(rho, z, mid_idx):
    """ Calculates distance between centers of mass """
    # Left
    rho_L = rho[:mid_idx]
    z_L = z[:mid_idx]
    norm_L = np.sum(rho_L)
    if norm_L < 1e-4: return 0
    com_L = np.sum(z_L * rho_L) / norm_L

    # Right
    rho_R = rho[mid_idx:]
    z_R = z[mid_idx:]
    norm_R = np.sum(rho_R)
    if norm_R < 1e-4: return 0
    com_R = np.sum(z_R * rho_R) / norm_R
    
    return abs(com_R - com_L)

def run_simulation(phase_diff):
    initial_dist = 5.0
    width = 1.2
    
    # Packet 1 (Left)
    psi1 = np.exp(-(z + initial_dist/2)**2 / (2*width**2))
    
    # Packet 2 (Right) with Phase Shift
    psi2 = np.exp(-(z - initial_dist/2)**2 / (2*width**2)) * np.exp(1j * phase_diff)
    
    Psi = psi1 + psi2
    # Normalize to keep nonlinearities in check
    Psi /= np.sqrt(np.sum(np.abs(Psi)**2) * dz) * 0.6 
    
    # Propagators
    k_mom = np.fft.fftfreq(N, d=dz) * 2 * np.pi
    Op_K = np.exp(-1j * (k_mom**2 / (2*mass)) * dt)
    
    history = []
    times = []
    mid_idx = N // 2
    
    t = 0
    steps = int(t_max / dt)
    
    for _ in range(steps):
        rho = np.abs(Psi)**2
        
        V_soler = -G_soler * rho / (1 + S_soler * rho)
        Phi = solve_poisson(rho)
        V_total = V_soler - xi * Phi
        
        sep = get_separation(rho, z, mid_idx)
        history.append(sep)
        times.append(t)
        
        # Step
        Op_V = np.exp(-1j * V_total * dt)
        Psi = Psi * Op_V
        Psi = np.fft.ifft(np.fft.fft(Psi) * Op_K)
        
        t += dt
        
    return np.array(times), np.array(history)

# --- MAIN ---

print("Simulating: Constructive Interference (In-Phase)...")
t1, sep1_raw = run_simulation(phase_diff=0.0)

print("Simulating: Destructive Interference (Anti-Phase)...")
t2, sep2_raw = run_simulation(phase_diff=np.pi)

# --- POST-PROCESSING ---
sigma_smooth = 20
sep1_smooth = gaussian_filter1d(sep1_raw, sigma=sigma_smooth)
sep2_smooth = gaussian_filter1d(sep2_raw, sigma=sigma_smooth)

# --- PLOTTING ---
plt.figure(figsize=(10, 6))

# Plot smoothed lines
plt.plot(t1, sep1_smooth, 'b-', linewidth=3.0, label=r'In-Phase ($\Delta\phi=0$)')
plt.plot(t2, sep2_smooth, 'r--', linewidth=3.0, label=r'Anti-Phase ($\Delta\phi=\pi$)')

# Styling
plt.title("Interference-Modulated Gravitational Interaction", fontweight='bold', fontsize=14)
plt.xlabel("Time [arbitrary units]", fontsize=12)
plt.ylabel("Soliton Separation", fontsize=12)
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(fontsize=11, loc='lower left', framealpha=0.9)

# Scientific Annotation
plt.text(t1[len(t1)//2], 4.6, 
         "Applicability: Coherent States\n(e.g. Boson Stars / SFDM)", 
         fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

# Arrows
mid_t = t1[len(t1)//2]
idx = len(t1)//2

plt.annotate("Enhanced Density\n(Attraction)", 
             xy=(mid_t, sep1_smooth[idx]), xytext=(mid_t, sep1_smooth[idx]-0.6),
             arrowprops=dict(facecolor='blue', shrink=0.05), ha='center', color='blue', fontweight='bold')

plt.annotate("Reduced Density\n(Suppression)", 
             xy=(mid_t, sep2_smooth[idx]), xytext=(mid_t, sep2_smooth[idx]+0.6),
             arrowprops=dict(facecolor='red', shrink=0.05), ha='center', color='red', fontweight='bold')

filename = "fig12_interference_modulated_gravity.png"
plt.savefig(filename, dpi=150, bbox_inches='tight')
print(f"Figure saved to {filename}")
plt.show()
