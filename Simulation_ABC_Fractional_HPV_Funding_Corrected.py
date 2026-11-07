import numpy as np
import matplotlib.pyplot as plt
from math import gamma
from scipy.optimize import root_scalar

# ------------------------------------------------------------------------------
# Parameters (stronger funding effect) -- funding in million USD
# ------------------------------------------------------------------------------
params = {
    'Pi': 186000.0,
    'beta': 0.5546,
    'mu': 0.1,
    'gamma': 0.005,
    'omega0': 0.4820,
    'sigma0': 0.9405,
    'delta0': 0.0500,
    'eps0': 0.0133,
    'rho': 0.25,
    'tau0': 0.9454,
    'alpha': 0.5013,
    'phi': 0.5926,
    'theta': 0.1,
    'kappa1': 0.16,
    'kappa2': 0.8383,
    'kappa3': 0.6427,
    'beta1': 0.02,
    'beta2': 0.001,
    'Lambda_F': 18.9,          # million USD per year
    'eta1': 1.98e-7,           # million USD per infected individual
    'eta2': 9.23e-6,           # million USD per infected individual
    'xi': 1.0,
}

# Initial conditions
S0 = 600000.0; V0 = 350000.0; Ia0 = 250000.0; As0 = 240000.0
Is0 = 220000.0; Ic0 = 200000.0; R0 = 0.0; F0 = 18.9          # million USD
C0 = 0.0
X0 = np.array([S0, V0, Ia0, As0, Is0, Ic0, R0, F0, C0])

nu = 0.9
Bnu = 1 - nu + nu / gamma(nu)
t_end = 30.0
h = 0.05
t = np.arange(0, t_end + h, h)

# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------
def rate_factor(F, params):
    """Funding multiplier m(F) = 1 + beta1*F / (1 + beta2*F)."""
    return 1.0 + params['beta1'] * F / (1.0 + params['beta2'] * F)

def capped_tau(F, params):
    """Capped screening proportion: tau(F) = min{1, tau0 * m(F)}.

    Enforces 0 <= tau(F) <= 1 for all F >= 0, as required because tau
    represents a biological proportion.
    """
    return min(1.0, params['tau0'] * rate_factor(F, params))

def compute_Re(F, params, beta=None):
    if beta is None:
        beta = params['beta']
    mu = params['mu']; alpha = params['alpha']; theta = params['theta']
    gamma_p = params['gamma']; phi = params['phi']
    kappa1 = params['kappa1']; kappa2 = params['kappa2']; kappa3 = params['kappa3']
    omega0 = params['omega0']; sigma0 = params['sigma0']
    delta0 = params['delta0']; eps0 = params['eps0']
    factor = rate_factor(F, params)
    omega = omega0 * factor
    tau = capped_tau(F, params)              # CAP: tau(F) = min(1, tau0*m(F))
    sigma = sigma0 * factor
    delta = delta0 * factor
    eps = eps0 * factor

    d1 = mu + alpha
    d2 = mu + sigma
    d3 = mu + delta + phi
    d4 = mu + gamma_p + eps

    factor_beta = beta * (theta + mu) / (theta + mu + omega)
    term1 = 1.0 / d1
    term2 = kappa1 * tau * alpha / (d1 * d2)
    term3 = kappa2 * (1 - tau) * alpha / (d1 * d3)
    term4 = kappa3 * phi * (1 - tau) * alpha / (d1 * d3 * d4)
    Re = factor_beta * (term1 + term2 + term3 + term4)
    return max(0.0, Re)

def derivatives(t, X, params, funding_active=True, fixed_F=None):
    S, V, Ia, As, Is, Ic, R, F, C = X
    N = S + V + Ia + As + Is + Ic + R
    lam = params['beta'] * (Ia + params['kappa1']*As + params['kappa2']*Is + params['kappa3']*Ic) / N
    if funding_active:
        f = F
    else:
        f = fixed_F if fixed_F is not None else 0.0
    factor = rate_factor(f, params)
    omega = params['omega0'] * factor
    tau = capped_tau(f, params)              # CAP: tau(F) = min(1, tau0*m(F))
    sigma = params['sigma0'] * factor
    delta = params['delta0'] * factor
    eps = params['eps0'] * factor

    dS = params['Pi'] - (lam + params['mu'] + omega) * S + params['rho'] * R + params['theta'] * V
    dV = omega * S - (params['theta'] + params['mu']) * V
    dIa = lam * S - (params['mu'] + params['alpha']) * Ia
    dAs = tau * params['alpha'] * Ia - (sigma + params['mu']) * As
    dIs = (1 - tau) * params['alpha'] * Ia - (params['mu'] + delta + params['phi']) * Is
    dIc = params['phi'] * Is - (params['mu'] + params['gamma'] + eps) * Ic
    dR = sigma * As + delta * Is + eps * Ic - (params['mu'] + params['rho']) * R
    dF = params['Lambda_F'] + params['eta1'] * Is + params['eta2'] * Ic - params['xi'] * F if funding_active else 0.0
    dC = lam * S
    return np.array([dS, dV, dIa, dAs, dIs, dIc, dR, dF, dC])

def solve_abc(system, X0, t, params, nu, funding_active=True, fixed_F=None):
    Nsteps = len(t) - 1
    h = t[1] - t[0]
    X = np.zeros((len(t), len(X0)))
    X[0] = X0
    Bnu = 1 - nu + nu / gamma(nu)

    for n in range(Nsteps):
        sumA = np.zeros(len(X0)); sumB = np.zeros(len(X0))
        for k in range(n + 1):
            A = (n + 1 - k)**nu * (n - k + 2 + nu) - (n - k)**nu * (n - k + 2 + 2*nu)
            sumA += A * system(t[k], X[k], params, funding_active, fixed_F)
        for k in range(1, n + 1):
            B = (n + 1 - k)**(nu + 1) - (n - k)**nu * (n - k + 1 + nu)
            sumB += B * system(t[k-1], X[k-1], params, funding_active, fixed_F)

        Fn = system(t[n], X[n], params, funding_active, fixed_F)
        Xp = X0 + (1 - nu) / Bnu * Fn + (nu * h**nu) / (Bnu * gamma(nu + 2)) * (sumA - sumB)

        Fn1 = system(t[n+1], Xp, params, funding_active, fixed_F)
        X_next = X0 + (1 - nu) / Bnu * Fn1 + (nu * h**nu) / (Bnu * gamma(nu + 2)) * (sumA - sumB + Fn1)
        X_next = np.maximum(X_next, 0.0)
        X[n+1] = X_next
    return X

# ------------------------------------------------------------------------------
# Set high beta and define funding levels (million USD)
# ------------------------------------------------------------------------------
beta_high = 2.5
params_high = params.copy()
params_high['beta'] = beta_high

def Re_minus_1(F):
    return compute_Re(F, params_high) - 1.0

Re0 = compute_Re(0.0, params_high)
Re_inf = compute_Re(1e6, params_high)

if Re0 > 1 and Re_inf < 1:
    sol = root_scalar(Re_minus_1, bracket=[0, 100], method='brentq')
    F_cross = sol.root
else:
    F_cross = 20.0

F_max = min(max(2 * F_cross, 40), 100)
F_levels = np.linspace(0, F_max, 6)
F_levels = np.round(F_levels, 1)

filtered_F = []
filtered_Re = []
for Fv in F_levels:
    re = compute_Re(Fv, params_high)
    if re >= 0:
        filtered_F.append(Fv)
        filtered_Re.append(re)
    else:
        print(f"Warning: Negative Re for F={Fv}, skipping.")
F_levels = np.array(filtered_F)
Re_vals = np.array(filtered_Re)

# ------------------------------------------------------------------------------
# Cap diagnostics
# ------------------------------------------------------------------------------
def F_crit(params):
    """Funding level at which the cap binds (None if it never binds)."""
    denom = params['tau0'] * params['beta1'] - (1.0 - params['tau0']) * params['beta2']
    if denom <= 0:
        return None
    return (1.0 - params['tau0']) / denom

Fc = F_crit(params_high)
print("=" * 74)
print("CAP STATUS: tau(F) = min{1, tau0 * m(F)} is ACTIVE in this run.")
print("=" * 74)
print(f"Funding levels (million USD): {F_levels}")
print(f"R_e values (capped): {np.round(Re_vals, 4)}")
print(f"Crossing funding F_cross (R_e=1) = {F_cross:.2f} million USD")
if Fc is None:
    print("Cap never binds (denominator <= 0).")
else:
    print(f"Cap binds for F >= F_crit = {Fc:.2f} million USD")
print()
print("Screening proportion at selected funding levels:")
print(f"  {'F':>6}  {'tau_uncapped':>14}  {'tau_capped':>10}  {'1 - tau':>8}")
print("  " + "-" * 46)
for Fv in [0.0, 10.0, 18.9, 20.0, 40.0]:
    tu = params_high['tau0'] * rate_factor(Fv, params_high)
    tc = capped_tau(Fv, params_high)
    print(f"  {Fv:6.1f}  {tu:14.4f}  {tc:10.4f}  {1.0 - tc:8.4f}")
print()

# ------------------------------------------------------------------------------
# Run simulations
# ------------------------------------------------------------------------------
sol_baseline_with = solve_abc(derivatives, X0, t, params, nu, funding_active=True)
sol_baseline_without = solve_abc(derivatives, X0, t, params, nu, funding_active=False, fixed_F=0.0)

sol_high_with = solve_abc(derivatives, X0, t, params_high, nu, funding_active=True)
sol_high_without = solve_abc(derivatives, X0, t, params_high, nu, funding_active=False, fixed_F=0.0)

solutions_F = {}
for Fv in F_levels:
    solutions_F[Fv] = solve_abc(derivatives, X0, t, params_high, nu,
                                funding_active=False, fixed_F=Fv)

idx_Ia, idx_As, idx_Is, idx_Ic = 2, 3, 4, 5
idx_F, idx_C = 7, 8

# ------------------------------------------------------------------------------
# Figure 1: 2x2 compartment shading (high beta)
# ------------------------------------------------------------------------------
compartments = [
    (idx_Ia, '$I_a$', 'tab:blue'),
    (idx_As, '$A_s$', 'tab:orange'),
    (idx_Is, '$I_s$', 'tab:green'),
    (idx_Ic, '$I_c$', 'tab:red')
]

fig1, axes1 = plt.subplots(2, 2, figsize=(12, 10))
axes1 = axes1.flatten()

for ax, (idx, label, color) in zip(axes1, compartments):
    with_f = sol_high_with[:, idx] / 1000
    without_f = sol_high_without[:, idx] / 1000
    ax.fill_between(t, without_f, with_f, color=color, alpha=0.3, label='Reduction')
    ax.plot(t, with_f, color=color, linestyle='-', linewidth=2, label='With funding')
    ax.plot(t, without_f, color=color, linestyle='--', linewidth=2, label='Without funding')
    ax.set_xlim(0, t_end)
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Population (thousands)')
    ax.set_title(f'{label} (high $\\beta$)')
    ax.legend(loc='best')
    ax.grid(True)

plt.tight_layout()
plt.savefig('compartment_shading.png', dpi=300)
plt.show()

# ------------------------------------------------------------------------------
# Figure 2: Total infected population at different funding levels (million USD)
# ------------------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(8, 5))
for Fv in F_levels:
    sol = solutions_F[Fv]
    total_inf = (sol[:, idx_Ia] + sol[:, idx_As] + sol[:, idx_Is] + sol[:, idx_Ic]) / 1000
    ax2.plot(t, total_inf, label=f'F = {Fv:.1f} million USD')
ax2.set_xlim(0, t_end)
ax2.set_xlabel('Time (years)')
ax2.set_ylabel('Total infected (thousands)')
ax2.set_title('Total infected population at various funding levels (high $\\beta$)')
ax2.legend()
ax2.grid(True)
plt.tight_layout()
plt.savefig('total_infected_funding_levels.png', dpi=300)
plt.show()

# ------------------------------------------------------------------------------
# Figure 3: Bar chart of Re vs funding level (million USD)
# ------------------------------------------------------------------------------
fig3, ax3 = plt.subplots(figsize=(7, 5))
bars = ax3.bar([f'{f:.1f}' for f in F_levels], Re_vals, color='skyblue', edgecolor='black')
ax3.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='$R_e=1$')
ax3.set_xlabel('Funding level $F$ (million USD)')
ax3.set_ylabel('Effective reproduction number $R_e$')
ax3.set_title('$R_e$ vs healthcare funding (high $\\beta$, capped $\\tau$)')
ax3.legend()
ax3.grid(axis='y', linestyle=':', alpha=0.7)
ax3.set_ylim(bottom=0)
for bar, val in zip(bars, Re_vals):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{val:.3f}', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig('Re_barchart.png', dpi=300)
plt.show()

# ------------------------------------------------------------------------------
# Figure 4: Total infected with/without funding with shading
# ------------------------------------------------------------------------------
fig4, ax4 = plt.subplots(figsize=(8, 5))
total_with = sol_high_with[:, idx_Ia] + sol_high_with[:, idx_As] + sol_high_with[:, idx_Is] + sol_high_with[:, idx_Ic]
total_without = sol_high_without[:, idx_Ia] + sol_high_without[:, idx_As] + sol_high_without[:, idx_Is] + sol_high_without[:, idx_Ic]
ax4.fill_between(t, total_without/1000, total_with/1000, color='lightblue', alpha=0.5, label='Reduction due to funding')
ax4.plot(t, total_with/1000, 'b-', linewidth=2, label='With funding')
ax4.plot(t, total_without/1000, 'r--', linewidth=2, label='Without funding')
ax4.set_xlim(0, t_end)
ax4.set_xlabel('Time (years)')
ax4.set_ylabel('Total infected (thousands)')
ax4.set_title('Total infected population: funding impact (high $\\beta$)')
ax4.legend()
ax4.grid(True)
plt.tight_layout()
plt.savefig('total_infected_shading.png', dpi=300)
plt.show()

# ------------------------------------------------------------------------------
# Figure 5: Contour plot
# ------------------------------------------------------------------------------
beta1_vals = np.linspace(0.001, 0.05, 30)
beta2_vals = np.linspace(0.0005, 0.005, 30)
B1, B2 = np.meshgrid(beta1_vals, beta2_vals)
Re_grid = np.zeros_like(B1)
params_temp = params.copy()
params_temp['beta'] = beta_high
for i in range(B1.shape[0]):
    for j in range(B1.shape[1]):
        params_temp['beta1'] = B1[i, j]
        params_temp['beta2'] = B2[i, j]
        Re_grid[i, j] = compute_Re(F0, params_temp)

fig5, ax5 = plt.subplots(figsize=(8, 6))
contour = ax5.contourf(B1, B2, Re_grid, levels=30, cmap='viridis')
ax5.contour(B1, B2, Re_grid, levels=[1.0], colors='red', linewidths=2, linestyles='dashed')
ax5.set_xlabel('$\\beta_1$')
ax5.set_ylabel('$\\beta_2$')
ax5.set_title('Contour of $R_e$ vs funding parameters at $F=F_0$ (high $\\beta$, capped $\\tau$)')
cbar = plt.colorbar(contour, ax=ax5)
cbar.set_label('$R_e$')
plt.tight_layout()
plt.savefig('contour_Re.png', dpi=300)
plt.show()

# ------------------------------------------------------------------------------
# Figure 6: Convergence plot
# ------------------------------------------------------------------------------
fig6, ax6 = plt.subplots(figsize=(8, 5))
total_high_with = sol_high_with[:, idx_Ia] + sol_high_with[:, idx_As] + sol_high_with[:, idx_Is] + sol_high_with[:, idx_Ic]
total_high_without = sol_high_without[:, idx_Ia] + sol_high_without[:, idx_As] + sol_high_without[:, idx_Is] + sol_high_without[:, idx_Ic]
total_baseline = sol_baseline_with[:, idx_Ia] + sol_baseline_with[:, idx_As] + sol_baseline_with[:, idx_Is] + sol_baseline_with[:, idx_Ic]

ax6.plot(t, total_baseline / 1000, 'g-', label='Baseline $\\beta$ (funding, $R_e<1$)')
ax6.plot(t, total_high_with / 1000, 'b-', label='High $\\beta$ with funding ($R_e\\approx 1.02$, $I_s^*=I_c^*=0$)')
ax6.plot(t, total_high_without / 1000, 'r--', label='High $\\beta$ without funding ($R_e>1$)')
ax6.set_xlim(0, t_end)
ax6.set_xlabel('Time (years)')
ax6.set_ylabel('Total infected (thousands)')
ax6.set_title('Convergence to disease-free equilibrium')
ax6.legend()
ax6.grid(True)
plt.tight_layout()
plt.savefig('convergence.png', dpi=300)
plt.show()

print("All simulations completed.")