import numpy as np
import matplotlib.pyplot as plt
from scipy.special import i0, i1, k0, k1

# =============================================================================
# CODE METADATA & PHYSICS ESSENCE
# =============================================================================
# SCRIPT: fig11_galactic_rotation_curves_v2.py
# PURPOSE: Validation of Scalar Gravity against Astrophysical Data (Corrected Fit).
#
# THEORETICAL BACKGROUND:
# We model the total rotation velocity v_tot(r) as the quadrature sum of:
# 1. Baryonic Disk (Visible Matter): Modeled via Freeman's analytic solution 
#    for a thin exponential disk.
#    v_disk^2(r) ~ (G M / Rd) * y^2 * [I0(y)K0(y) - I1(y)K1(y)], with y = r/2Rd.
#
# 2. Scalar Halo (Dark Matter proxy): Modeled as a Pseudo-Isothermal Sphere,
#    arising from the non-singular core of the scalar field soliton.
#    v_scalar^2(r) = V_inf^2 * (1 - (Rs/r)*arctan(r/Rs)).
#
# ADJUSTMENTS V2:
# Corrected the mass normalization factors (M_scale) for the Freeman disk
# to properly reproduce the magnitude of the stellar component (~200 km/s 
# for massive spirals) seen in the preprint.
# =============================================================================

# --- CONFIGURATION ---

class GalaxyData:
    def __init__(self, name, r_data, v_data, v_err, params):
        self.name = name
        self.r_data = np.array(r_data)
        self.v_data = np.array(v_data)
        self.v_err = np.array(v_err)
        self.Rd = params['Rd']       # Disk Scale Length [kpc]
        self.Rs = params['Rs']       # Scalar Halo Scale Length [kpc]
        self.M_scale = params['Md']  # Mass Normalization Factor (G*M/L units)
        self.V_inf = params['Vh']    # Asymptotic Halo Velocity [km/s]

# --- DATA & PARAMETERS ---
# Parameters tuned to match the visual fit of the preprint
# (Ensuring "Blue" disk curve has correct amplitude)

# 1. NGC 6503 (Spiral)
g1 = GalaxyData(
    name="NGC 6503 (Spiral)",
    r_data=[0.5, 1.9, 2.8, 3.7, 4.6, 5.5, 6.4, 7.3, 8.2, 9.2, 11.0, 12.8, 14.6, 18.3, 22.0],
    v_data=[25.0, 82.0, 104.0, 113.0, 115.0, 116.0, 117.0, 117.5, 118.0, 118.2, 118.5, 119.0, 119.2, 119.5, 119.8],
    v_err=[5]*15,
    # Needs strong disk component to explain the sharp rise
    params={'Rd': 1.3, 'Rs': 1.1, 'Md': 22000.0, 'Vh': 120.0} 
)

# 2. NGC 2841 (Massive Spiral)
g2 = GalaxyData(
    name="NGC 2841 (Massive)",
    r_data=[1.5, 3.5, 5.5, 8.0, 11.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0],
    v_data=[150.0, 260.0, 295.0, 308.0, 310.0, 300.0, 290.0, 280.0, 278.0, 270.0, 265.0],
    v_err=[8]*11,
    # Massive disk required to hit the ~300 km/s peak at short radii
    params={'Rd': 3.4, 'Rs': 1.5, 'Md': 240000.0, 'Vh': 280.0}
)

# 3. DDO 154 (Dwarf)
g3 = GalaxyData(
    name="DDO 154 (Dwarf)",
    r_data=[0.4, 0.8, 1.5, 2.2, 3.0, 4.0, 5.0, 6.0, 7.0],
    v_data=[10.0, 25.0, 35.0, 43.0, 47.0, 49.0, 50.0, 51.0, 52.0],
    v_err=[3]*9,
    # Dominant scalar halo, small disk component
    params={'Rd': 1.0, 'Rs': 2.5, 'Md': 7000.0, 'Vh': 65.0}
)

galaxies = [g1, g2, g3]

# --- PHYSICS MODELS ---

def velocity_disk_freeman(r, Rd, M_scale):
    """
    Computes rotation velocity of an exponential disk (Freeman 1970).
    """
    r_safe = np.maximum(r, 1e-3)
    y = r_safe / (2 * Rd)
    
    # Bessel terms: I0*K0 - I1*K1
    # The function has a maximum around y ~ 1.1
    bessel_term = i0(y) * k0(y) - i1(y) * k1(y)
    
    # The term (y**2) handles the r -> 0 behavior
    v2 = M_scale * (y**2) * bessel_term
    return np.sqrt(np.maximum(v2, 0))

def velocity_scalar_halo(r, Rs, V_inf):
    """
    Computes velocity for a Pseudo-Isothermal Sphere (Scalar Halo).
    """
    r_safe = np.maximum(r, 1e-3)
    x = r_safe / Rs
    # v^2 = V_inf^2 * (1 - (1/x)*arctan(x))
    v2 = V_inf**2 * (1 - (1/x) * np.arctan(x))
    return np.sqrt(np.maximum(v2, 0))

# --- PLOTTING ---

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
plt.subplots_adjust(wspace=0.2, left=0.05, right=0.95)

for i, gal in enumerate(galaxies):
    ax = axes[i]
    
    # Radius grid for smooth lines
    r_max = max(gal.r_data) * 1.1
    r_smooth = np.linspace(0.1, r_max, 200)
    
    # 1. Calculate Models
    v_disk = velocity_disk_freeman(r_smooth, gal.Rd, gal.M_scale)
    v_halo = velocity_scalar_halo(r_smooth, gal.Rs, gal.V_inf)
    v_total = np.sqrt(v_disk**2 + v_halo**2)
    
    # 2. Plot Data
    ax.errorbar(gal.r_data, gal.v_data, yerr=gal.v_err, fmt='o', color='k', 
                markersize=4, capsize=2, elinewidth=1, alpha=0.7, label='Data')
    
    # 3. Plot Models
    # Blue dashed: Baryons
    ax.plot(r_smooth, v_disk, color='blue', linestyle='--', linewidth=1.5, 
            label='Baryon Disk (3D)')
    
    # Green dotted: Scalar Halo
    ax.plot(r_smooth, v_halo, color='green', linestyle=':', linewidth=2.0, 
            label='Scalar Vacuum')
    
    # Red solid: Total
    ax.plot(r_smooth, v_total, color='red', linestyle='-', linewidth=2.5, 
            label='Total Model')
    
    # Formatting
    ax.set_title(gal.name, fontweight='bold', fontsize=11)
    ax.set_xlabel('Radius [kpc]')
    
    if i == 0:
        ax.set_ylabel('Velocity [km/s]')
    
    ax.grid(True, alpha=0.3, which='both')
    ax.minorticks_on()
    
    # Parameter Box (Fixed positions per galaxy for readability)
    textstr = '\n'.join((
        r'$R_d=%.1f$ kpc' % (gal.Rd, ),
        r'$R_s=%.1f$ kpc' % (gal.Rs, )))
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='lightgray')
    
    # Position similar to preprint
    ax.text(0.55, 0.15, textstr, transform=ax.transAxes, fontsize=9, bbox=props)
    
    # Legend only on first plot or reduced
    if i == 0:
        ax.legend(loc='center right', fontsize=9)
    else:
        # Simplified legend or copy
        ax.legend(loc='center right', fontsize=8)

# Global Title
plt.suptitle("Figure 11: Galactic Rotation Curves. 3+1D Scalar Gravity Fit.", 
             fontsize=14, fontweight='bold', y=0.98)

filename = "fig11_galactic_rotation_curves.png"
plt.savefig(filename, dpi=150, bbox_inches='tight')
print(f"Figure saved to {filename}")
plt.show()
