"""
Full solution script for DME equilibrium assignment.
This is the complete, runnable code for the Jupyter notebook.
"""
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ===========================================================
# 1. THERMODYNAMIC DATA (NIST Shomate equations, 298–1000 K)
# ===========================================================
# Components: CO2, H2, CH3OH (methanol), H2O, CO, CH3OCH3 (DME)
# Shomate parameters [A, B, C, D, E, F, G, H_ref]
# H_ref is set so that H(T)-H(298.15K) = 0 at T = 298.15 K
# Standard formation enthalpies dHf298 are stored separately

_p_dme = [13.88, 170.2, -103.3, 27.35, 0.1152, -238.8, 265.8]
_t0 = 298.15 / 1000
_H_ref_dme = sum([_p_dme[0]*_t0, _p_dme[1]*_t0**2/2, _p_dme[2]*_t0**3/3,
                  _p_dme[3]*_t0**4/4, -_p_dme[4]/_t0, _p_dme[5]])

shomate = {
    # NIST WebBook: https://webbook.nist.gov/chemistry/
    'CO2':   [24.99735, 55.18696, -33.69137,  7.948387, -0.136638, -403.6075, 228.2431, -393.5224],
    'H2':    [33.06618, -11.36342,  11.43282, -2.772874, -0.158558,  -9.98080, 172.7080,    0.0   ],
    'CH3OH': [14.19520,  97.72180,  -9.73000,-12.81940,  0.158800, -209.7085, 216.1400, -200.9400],
    'H2O':   [30.09200,   6.83251,   6.79344, -2.534480,  0.082139, -250.8810, 223.3967, -241.8264],
    'CO':    [25.56759,   6.09613,   4.05466, -2.671301,  0.131021, -118.0089, 227.3665, -110.5271],
    'DME':   _p_dme + [_H_ref_dme],  # NASA / NIST (230K–1000K)
}

# Standard formation enthalpies [kJ/mol] at 298.15 K (NIST)
dHf298 = {
    'CO2': -393.51, 'H2': 0.0, 'CH3OH': -200.94,
    'H2O': -241.826, 'CO': -110.527, 'DME': -184.1,
}

COMPS = ['CO2', 'H2', 'CH3OH', 'H2O', 'CO', 'DME']
R_gas = 8.3145  # J / (mol K)

def prop_thermo(T, comp):
    """
    Returns [H_f (J/mol), S (J/mol/K)] at temperature T (K)
    using the NIST Shomate equations.
    """
    p = np.array(shomate[comp], dtype=float)
    t = T / 1000.0
    # Enthalpy relative to 298.15 K [kJ/mol], then add ΔfH°(298.15)
    H_diff = (p[0]*t + p[1]*t**2/2 + p[2]*t**3/3 + p[3]*t**4/4
              - p[4]/t + p[5] - p[7]) * 1000    # J/mol
    H_abs = dHf298[comp] * 1000 + H_diff         # J/mol, absolute formation enthalpy
    # Entropy [J/mol/K]
    S = (p[0]*np.log(t) + p[1]*t + p[2]*t**2/2
         + p[3]*t**3/3 - p[4]*t**-2/2 + p[6])
    return np.array([H_abs, S])

# --- Validation at 298.15 K ---
print("=== Validation: ΔfH° at 298.15 K [kJ/mol] ===")
for c in COMPS:
    H, S = prop_thermo(298.15, c)
    print(f"  {c:<8}: H_calc={H/1000:.2f}  H_lit={dHf298[c]:.2f}")

# --- Reaction enthalpies at 298.15 K ---
nu_all = np.array([
    [-1., -3.,  1.,  1.,  0.,  0.],  # R1: CO2 + 3H2 <-> CH3OH + H2O
    [ 0., -2.,  1.,  0., -1.,  0.],  # R2: CO  + 2H2 <-> CH3OH
    [-1., -1.,  0.,  1.,  1.,  0.],  # R3: CO2 +  H2 <-> CO   + H2O (rWGS)
    [ 0.,  0., -2.,  1.,  0.,  1.],  # R4: 2 CH3OH   <-> DME  + H2O
])
thermo298 = np.array([prop_thermo(298.15, c) for c in COMPS])
print("\n=== Reaction enthalpies at 298.15 K ===")
rxn_names = ['R1 (CO2+3H2→CH3OH+H2O)', 'R2 (CO+2H2→CH3OH)',
             'R3 (CO2+H2→CO+H2O)', 'R4 (2CH3OH→DME+H2O)']
for i, name in enumerate(rxn_names):
    dH = nu_all[i] @ thermo298[:,0] / 1000
    dS = nu_all[i] @ thermo298[:,1]
    dG = dH*1000 - 298.15*dS
    K  = np.exp(-dG / (R_gas*298.15))
    print(f"  {name}: ΔH={dH:.1f} kJ/mol, K°={K:.3g}")

# ===========================================================
# 2. STOICHIOMETRIC MATRIX & RANK ANALYSIS
# ===========================================================
print("\n=== Stoichiometric matrix rank ===")
rank = np.linalg.matrix_rank(nu_all)
print(f"  Rank = {rank}")
print(f"  → Only {rank} of the 4 reactions are linearly independent.")
print(f"    R1 = R2 + R3, so R1 is linearly dependent on R2 and R3.")
print(f"    All 3 independent reactions (R2, R3, R4) are used for the equilibrium calculation.")

# Use 3 independent reactions for the Gibbs minimisation
nu_ind = np.array([
    [-1., -3.,  1.,  1.,  0.,  0.],  # R1
    [-1., -1.,  0.,  1.,  1.,  0.],  # R3
    [ 0.,  0., -2.,  1.,  0.,  1.],  # R4
])

# ===========================================================
# 3. GIBBS FREE ENERGY MINIMISATION
# ===========================================================
def get_muRT0(T):
    """Standard chemical potential / RT = ΔfG°(T) / (RT) for each component."""
    muRT0 = np.zeros(6)
    for i, c in enumerate(COMPS):
        H, S = prop_thermo(T, c)
        muRT0[i] = (H - T*S) / (R_gas*T)
    return muRT0

def solve_gibbs(n_in, T, p):
    """
    Minimise total Gibbs free energy over 3 independent reaction extents xi.
    n_in : array(6), inlet molar flow rates [mol/s]
    T    : temperature [K]
    p    : pressure [bar]
    Returns n_out array(6) or None if no physical solution found.
    """
    muRT0 = get_muRT0(T)
    p0 = 1.0  # bar (reference pressure)

    def gibbs(xi):
        n = n_in + nu_ind.T @ xi
        if np.any(n < 0):
            return 1e10
        n = np.maximum(n, 1e-15)
        n_tot = np.sum(n)
        x = n / n_tot
        # G/RT = Σ n_i [μ°_i/RT + ln(x_i) + ln(p/p°)]
        return np.sum(n * (muRT0 + np.log(x) + np.log(p / p0)))

    n_sum = np.sum(n_in)
    best = None
    for xi1 in np.linspace(0.02, 0.9, 6):
        for xi3 in np.linspace(-0.4, 0.2, 4):
            for xi4 in np.linspace(0.0, 0.4, 4):
                xi0 = np.array([xi1, xi3, xi4]) * n_sum
                if np.any(n_in + nu_ind.T @ xi0 < 0):
                    continue
                res = minimize(gibbs, xi0, method='Nelder-Mead',
                               options={'xatol': 1e-12, 'fatol': 1e-12,
                                        'maxiter': 100000, 'adaptive': True})
                n_out = n_in + nu_ind.T @ res.x
                if np.all(n_out >= -1e-3) and res.fun < 1e9:
                    n_out = np.maximum(n_out, 0)
                    if best is None or res.fun < best[0]:
                        best = (res.fun, n_out)
    return best[1] if best else None

# ===========================================================
# 4. SINGLE OPERATING POINT (detailed output)
# ===========================================================
n_in_base = np.array([1.0, 3.0, 0.0, 0.0, 0.0, 0.0])  # CO2, H2, CH3OH, H2O, CO, DME [mol/s]
T_example = 250 + 273.15   # K
p_example = 50             # bar

n_out_ex = solve_gibbs(n_in_base, T_example, p_example)
x_out_ex = n_out_ex / np.sum(n_out_ex)

print(f"\n=== Operating point: T = {int(T_example-273.15)} °C, p = {p_example} bar ===")
print(f"    Inlet: CO₂=1, H₂=3 mol/s")
print(f"{'Component':<10} {'n_in':>8} {'n_out':>8} {'x_out':>8}")
for i, c in enumerate(COMPS):
    print(f"  {c:<8} {n_in_base[i]:>8.3f} {n_out_ex[i]:>8.4f} {x_out_ex[i]:>8.4f}")
Y_DME_ex = n_out_ex[5] / (n_in_base[0] / 2)
X_CO2_ex = (n_in_base[0] - n_out_ex[0]) / n_in_base[0]
print(f"\n  DME equilibrium yield   Y_DME  = {Y_DME_ex:.4f}")
print(f"  CO₂ equilibrium conversion X_CO₂ = {X_CO2_ex:.4f}")

# ===========================================================
# 5. PARAMETRIC STUDY: Temperature & Pressure
# ===========================================================
T_arr = np.linspace(150+273.15, 450+273.15, 25)
p_arr = np.array([20., 50., 100.])

Y_DME  = np.full((3, len(T_arr)), np.nan)
X_CO2  = np.full((3, len(T_arr)), np.nan)
Y_MeOH = np.full((3, len(T_arr)), np.nan)

print("\nComputing parametric study (T, p)...")
for pi, p in enumerate(p_arr):
    for ti, T in enumerate(T_arr):
        n_out = solve_gibbs(n_in_base, T, p)
        if n_out is not None:
            Y_DME[pi,ti]  = n_out[5] / (n_in_base[0] / 2)
            X_CO2[pi,ti]  = (n_in_base[0] - n_out[0]) / n_in_base[0]
            Y_MeOH[pi,ti] = n_out[2] / n_in_base[0]

# ===========================================================
# 6. PARAMETRIC STUDY: H2/CO2 ratio
# ===========================================================
ratios = np.array([2.0, 2.5, 3.0, 3.5, 4.0, 5.0])
T_ratio = 250 + 273.15
p_ratio = 50.

Y_DME_r = np.full(len(ratios), np.nan)
X_CO2_r = np.full(len(ratios), np.nan)

print("Computing parametric study (H2/CO2 ratio)...")
for ri, r in enumerate(ratios):
    n_in = np.array([1.0, r, 0., 0., 0., 0.])
    n_out = solve_gibbs(n_in, T_ratio, p_ratio)
    if n_out is not None:
        Y_DME_r[ri] = n_out[5] / (n_in[0] / 2)
        X_CO2_r[ri] = (n_in[0] - n_out[0]) / n_in[0]

# ===========================================================
# 7. PLOTS
# ===========================================================
colors = ['#e74c3c', '#27ae60', '#2980b9']
T_C = T_arr - 273.15

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# (a) DME yield vs T
ax = axes[0, 0]
for pi, p in enumerate(p_arr):
    ax.plot(T_C, Y_DME[pi], color=colors[pi], lw=2.5, label=f'{int(p)} bar')
ax.set_xlabel('T / °C', fontsize=12); ax.set_ylabel('Y$_{DME}$ / –', fontsize=12)
ax.set_title('(a) DME equilibrium yield vs. temperature', fontsize=11)
ax.legend(fontsize=11); ax.grid(True, alpha=0.4)
ax.set_xlim([150, 450]); ax.set_ylim([0, 1])

# (b) CO2 conversion vs T
ax = axes[0, 1]
for pi, p in enumerate(p_arr):
    ax.plot(T_C, X_CO2[pi], color=colors[pi], lw=2.5, label=f'{int(p)} bar')
ax.set_xlabel('T / °C', fontsize=12); ax.set_ylabel('X$_{CO_2}$ / –', fontsize=12)
ax.set_title('(b) CO₂ equilibrium conversion vs. temperature', fontsize=11)
ax.legend(fontsize=11); ax.grid(True, alpha=0.4)
ax.set_xlim([150, 450]); ax.set_ylim([0, 1])

# (c) Methanol yield vs T
ax = axes[1, 0]
for pi, p in enumerate(p_arr):
    ax.plot(T_C, Y_MeOH[pi], color=colors[pi], lw=2.5, label=f'{int(p)} bar')
ax.set_xlabel('T / °C', fontsize=12); ax.set_ylabel('Y$_{CH_3OH}$ / –', fontsize=12)
ax.set_title('(c) Methanol equilibrium yield (intermediate)', fontsize=11)
ax.legend(fontsize=11); ax.grid(True, alpha=0.4)
ax.set_xlim([150, 450])

# (d) DME yield & CO2 conv vs H2/CO2 ratio
ax = axes[1, 1]
ax.plot(ratios, Y_DME_r, 'o-', color='#2980b9', lw=2.5, ms=8, label='Y$_{DME}$')
ax.plot(ratios, X_CO2_r, 's--', color='#e74c3c', lw=2.5, ms=8, label='X$_{CO_2}$')
ax.axvline(x=3.0, color='grey', ls=':', lw=1.5, label='Stoich. H₂/CO₂ = 3')
ax.set_xlabel('H₂/CO₂ molar ratio', fontsize=12)
ax.set_ylabel('Yield / Conversion / –', fontsize=12)
ax.set_title(f'(d) Effect of H₂/CO₂ ratio  (T = {int(T_ratio-273.15)} °C, p = {int(p_ratio)} bar)', fontsize=11)
ax.legend(fontsize=11); ax.grid(True, alpha=0.4); ax.set_ylim([0, 1])

plt.suptitle('DME Synthesis — Chemical Equilibrium Analysis', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/dme_equilibrium.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved.")

# ===========================================================
# 8. SUMMARY TABLE
# ===========================================================
print("\n=== DME yield, CO2 conversion and MeOH yield at selected operating points ===")
print(f"{'T/°C':<8} {'p/bar':<8} {'Y_DME':<10} {'X_CO2':<10} {'Y_MeOH':<10}")
for T_val in [200, 250, 300, 350, 400]:
    ti = np.argmin(np.abs(T_arr - (T_val+273.15)))
    for pi, p in enumerate(p_arr):
        s = f"  {T_val:<6} {int(p):<8}"
        for val in [Y_DME[pi,ti], X_CO2[pi,ti], Y_MeOH[pi,ti]]:
            s += f" {val:.4f}   "
        print(s)
    print()

print("\n=== DME yield vs H2/CO2 ratio (T=250°C, p=50 bar) ===")
for ri, r in enumerate(ratios):
    print(f"  H2/CO2 = {r:.1f}:  Y_DME = {Y_DME_r[ri]:.4f},  X_CO2 = {X_CO2_r[ri]:.4f}")
