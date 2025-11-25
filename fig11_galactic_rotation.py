import numpy as np
import matplotlib.pyplot as plt
from scipy.special import i0, i1, k0, k1
from scipy.optimize import curve_fit

# =============================================================================
# CODE METADATA & PHYSICS ESSENCE 
# =============================================================================
# SCRIPT: fig11_galactic_rotation.py
# PURPOSE: Macroscopic validation via Galactic Rotation Curve Decomposition.
#
# THEORETICAL BACKGROUND:
# On cosmological scales, the massless scalar field \Phi acts as a superfluid 
# background (Scalar Field Dark Matter / SFDM). The gradient of this potential 
# generates an effective centripetal acceleration:
#
#     v_{rot}(r) = \sqrt{r * d\Phi/dr}
#
# Unlike Cold Dark Matter (CDM) which predicts cuspy cores, the scalar field 
# naturally forms cored density profiles due to quantum pressure (Heisenberg 
# uncertainty preventing collapse at r->0).
#
# NUMERICAL METHOD:
# We perform a Non-Linear Least Squares (Levenberg-Marquardt) optimization to
# decompose observational data into two components:
# 1. Baryonic Disk: Fixed scale length R_d (Freeman Exponential Disk).
# 2. Scalar Halo: Pseudo-Isothermal profile characteristic of scalar cores.
#
# The minimization of the reduced Chi-Squared statistic (\chi^2_nu) confirms 
# that the emergent scalar gravity is consistent with observed flat rotation curves.
# =============================================================================

# --- PHYSICS KERNELS ---

def velocity_disk_freeman(r, Rd, M_scale):
    """
    Freeman (1970) exponential disk velocity.
    Represents the contribution of visible baryonic matter (Stars/Gas).
    """
    r = np.asarray(r)
    r_safe = np.maximum(r, 1e-3)
    y = r_safe / (2 * Rd)
    # Evaluation involves modified Bessel functions
    bessel_term = i0(y) * k0(y) - i1(y) * k1(y)
    v2 = M_scale * (y**2) * bessel_term
    return np.sqrt(np.maximum(v2, 0))

def velocity_scalar_halo(r, Rs, V_inf):
    """
    Pseudo-Isothermal Sphere velocity (Scalar Field Core).
    Represents the effective potential well created by the scalar field \Phi.
    """
    r = np.asarray(r)
    r_safe = np.maximum(r, 1e-3)
    x = r_safe / Rs
    # Cored profile (finite density at center), unlike NFW
    v2 = V_inf**2 * (1 - (1/x) * np.arctan(x))
    return np.sqrt(np.maximum(v2, 0))

# --- DATA STRUCTURE ---

class GalaxyFit:
    def __init__(self, name, r_data, v_data, v_err, Rd_fixed, initial_guess):
        self.name = name
        self.r_data = np.array(r_data)
        self.v_data = np.array(v_data)
        self.v_err = np.array(v_err)
        self.Rd = Rd_fixed # Photometric scale length (fixed from optical observations)
        
        # Initial guess for optimizer: [M_scale, Rs, V_inf]
        self.p0 = initial_guess 
        self.popt = None # Will hold optimized parameters
        self.perr = None # Will hold parameter errors
        self.chi2_red = 0 # Reduced Chi-Squared

    def fit_model(self):
        """
        Performs the Least Squares fit to separate Baryonic vs Scalar contributions.
        """
        # Define the total velocity function wrapper for curve_fit
        # We freeze Rd (baryonic distribution), so the optimizer only tunes the mass and halo
        def model_total_v(r, m_scale, r_s, v_inf):
            v_d = velocity_disk_freeman(r, self.Rd, m_scale)
            v_h = velocity_scalar_halo(r, r_s, v_inf)
            return np.sqrt(v_d**2 + v_h**2)

        # Bounds: Mass > 0, Radius > 0.1, Velocity > 0
        bounds = ([1.0, 0.1, 10.0], [np.inf, 50.0, 1000.0])

        # Run Optimization
        popt, pcov = curve_fit(
            model_total_v, 
            self.r_data, 
            self.v_data, 
            p0=self.p0, 
            sigma=self.v_err, 
            absolute_sigma=True,
            bounds=bounds,
            maxfev=5000
        )
        
        self.popt = popt
        self.perr = np.sqrt(np.diag(pcov))
        
        # Calculate Chi-Squared
        residuals = self.v_data - model_total_v(self.r_data, *popt)
        chi2 = np.sum((residuals / self.v_err)**2)
        dof = len(self.v_data) - len(popt) # Degrees of Freedom
        self.chi2_red = chi2 / dof
        
        print(f"--- {self.name} ---")
        print(f"Fit Results: M_scale={popt[0]:.1e}, Rs={popt[1]:.2f}, V_inf={popt[2]:.1f}")
        print(f"Reduced Chi-Square: {self.chi2_red:.3f}")

# --- INITIALIZATION ---

# 1. NGC 6503
g1 = GalaxyFit(
    "NGC 6503 (Spiral)",
    r_data=[0.5, 1.9, 2.8, 3.7, 4.6, 5.5, 6.4, 7.3, 8.2, 9.2, 11.0, 12.8, 14.6, 18.3, 22.0],
    v_data=[25.0, 82.0, 104.0, 113.0, 115.0, 116.0, 117.0, 117.5, 118.0, 118.2, 118.5, 119.0, 119.2, 119.5, 119.8],
    v_err=[5.0]*15,
    Rd_fixed=1.3,
    initial_guess=[22000, 1.1, 120] 
)

# 2. NGC 2841
g2 = GalaxyFit(
    "NGC 2841 (Massive)",
    r_data=[1.5, 3.5, 5.5, 8.0, 11.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0],
    v_data=[150.0, 260.0, 295.0, 308.0, 310.0, 300.0, 290.0, 280.0, 278.0, 270.0, 265.0],
    v_err=[8.0]*11,
    Rd_fixed=3.4,
    initial_guess=[240000, 1.5, 280]
)

# 3. DDO 154
g3 = GalaxyFit(
    "DDO 154 (Dwarf)",
    r_data=[0.4, 0.8, 1.5, 2.2, 3.0, 4.0, 5.0, 6.0, 7.0],
    v_data=[10.0, 25.0, 35.0, 43.0, 47.0, 49.0, 50.0, 51.0, 52.0],
    v_err=[3.0]*9,
    Rd_fixed=1.0,
    initial_guess=[7000, 2.5, 65]
)

galaxies = [g1, g2, g3]

# Perform Fits
for g in galaxies:
    g.fit_model()

# --- PLOTTING ---

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
plt.subplots_adjust(wspace=0.2, left=0.05, right=0.95)

for i, gal in enumerate(galaxies):
    ax = axes[i]
    
    # Unpack fitted parameters
    M_opt, Rs_opt, V_opt = gal.popt
    
    # Generate smooth curves
    r_smooth = np.linspace(0.1, max(gal.r_data)*1.1, 200)
    
    v_disk = velocity_disk_freeman(r_smooth, gal.Rd, M_opt)
    v_halo = velocity_scalar_halo(r_smooth, Rs_opt, V_opt)
    v_total = np.sqrt(v_disk**2 + v_halo**2)
    
    # Plot Data with Errors
    ax.errorbar(gal.r_data, gal.v_data, yerr=gal.v_err, fmt='o', color='k', 
                markersize=4, capsize=2, alpha=0.6, label='Data')
    
    # Plot Fitted Components
    ax.plot(r_smooth, v_disk, color='blue', linestyle='--', linewidth=1.5, 
            label='Baryon Disk (Fit)')
    ax.plot(r_smooth, v_halo, color='green', linestyle=':', linewidth=2.0, 
            label='Scalar Vacuum (Fit)')
    ax.plot(r_smooth, v_total, color='red', linestyle='-', linewidth=2.5, 
            label='Total Best Fit')
    
    # Styling
    ax.set_title(gal.name, fontweight='bold', fontsize=11)
    ax.set_xlabel('Radius [kpc]')
    if i == 0: ax.set_ylabel('Velocity [km/s]')
    ax.grid(True, alpha=0.3, which='both')
    
    # Stats Box
    # Showing fitted values confirms we aren't just guessing
    stats_text = (
        r'$R_d$ (fix) = ' + f'{gal.Rd} kpc\n' +
        r'$R_s$ (fit) = ' + f'{Rs_opt:.2f} kpc\n' + 
        r'$\chi^2_{\nu}$ = ' + f'{gal.chi2_red:.2f}'
    )
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    ax.text(0.55, 0.12, stats_text, transform=ax.transAxes, fontsize=9, bbox=props)

    if i == 0: ax.legend(loc='center right', fontsize=9)

plt.suptitle("Figure 11: Galactic Rotation Curves. Least-Squares Fit of Scalar Gravity Model.", 
             fontsize=14, fontweight='bold', y=0.98)

filename = "fig11_galactic_rotation_fit_optimized.png"
plt.savefig(filename, dpi=150, bbox_inches='tight')
print(f"Figure saved to {filename}")
plt.show()
