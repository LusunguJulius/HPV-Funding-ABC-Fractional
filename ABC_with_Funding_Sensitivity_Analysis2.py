import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import gamma
from scipy import stats
from scipy.stats import qmc

plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.default'] = 'it'

trapezoid = getattr(np, "trapezoid", None) or np.trapz

# ------------------------------------------------------------
# 1. PARAMETER TABLE: baseline, range of variation, distribution, basis
#    (this single table drives the sampling AND the LaTeX table,
#     so the ranges reported in the paper are exactly those used)
# ------------------------------------------------------------
DIST = "Uniform"          # distribution used for LHS sampling
REL = (0.7, 1.3)          # default: +/-30% of baseline
REL_FUND = (0.5, 2.0)     # funding parameters: 50%-200% of baseline

BASIS_DEFAULT = r"$\pm 30\%$ of baseline (assumed)"
BASIS_FUND = r"$50\%$--$200\%$ of baseline (new parameter, wider uncertainty)"


def rel(b, f):
    return b * f[0], b * f[1]


# (symbol, description, baseline, (lower, upper), basis)
PARAMS = [
    (r'\beta',        'Transmission rate',                       0.5546,  rel(0.5546, REL),  BASIS_DEFAULT),
    (r'\mu',          'Natural death rate',                      0.1,     rel(0.1, REL),     BASIS_DEFAULT),
    (r'\alpha',       'Progression rate from acute infection',   0.5013,  rel(0.5013, REL),  BASIS_DEFAULT),
    (r'\sigma',       'Clearance rate (asymptomatic)',           0.9405,  rel(0.9405, REL),  BASIS_DEFAULT),
    (r'\delta',       'Recovery rate (symptomatic)',             0.0500,  rel(0.05, REL),    BASIS_DEFAULT),
    (r'\phi',         'Progression rate to cancer',              0.5926,  rel(0.5926, REL),  BASIS_DEFAULT),
    (r'\gamma',       'Cancer-induced death rate',               0.005,   rel(0.005, REL),   BASIS_DEFAULT),
    (r'\varepsilon',  'Cancer treatment/recovery rate',          0.0133,  rel(0.0133, REL),  BASIS_DEFAULT),
    (r'\rho',         'Immunity waning rate',                    0.25,    rel(0.25, REL),    BASIS_DEFAULT),
    (r'\omega',       'Vaccination rate',                        0.4820,  rel(0.482, REL),   BASIS_DEFAULT),
    (r'\theta',       'Vaccine waning rate',                     0.1,     rel(0.1, REL),     BASIS_DEFAULT),
    (r'\tau',         'Proportion becoming asymptomatic',        0.9454,  (0.7, 1.0),        r"Proportion, bounded to $[0.7,\,1]$"),
    (r'\kappa_1',     'Relative infectiousness (As)',            0.16,    (0.1, 0.3),        r"Modification factor, bounded to $[0.1,\,0.3]$"),
    (r'\kappa_2',     'Relative infectiousness (Is)',            0.8383,  (0.5, 0.95),       r"Modification factor, bounded to $[0.5,\,0.95]$"),
    (r'\kappa_3',     'Relative infectiousness (Ic)',            0.6427,  (0.5, 0.9),        r"Modification factor, bounded to $[0.5,\,0.9]$"),
    (r'\Pi',          'Recruitment rate',                        186000,  rel(186000, REL),  BASIS_DEFAULT),
    (r'\nu',          'Fractional order',                        0.95,    (0.8, 1.0),        r"Admissible order, $0.8\le\nu\le 1$"),
    (r'\beta_1',      'Funding effect (saturation gain)',        2.57e-4, rel(2.57e-4, REL_FUND), BASIS_FUND),
    (r'\beta_2',      'Funding effect (saturation rate)',        5.29e-2, rel(5.29e-2, REL_FUND), BASIS_FUND),
    (r'\Lambda_F',    'Funding inflow rate',                     18.9,    rel(18.9, REL_FUND),    BASIS_FUND),
    (r'\eta_1',       'Funding contribution from Is',            1.98e-7, rel(1.98e-7, REL_FUND), BASIS_FUND),
    (r'\eta_2',       'Funding contribution from Ic',            9.23e-6, rel(9.23e-6, REL_FUND), BASIS_FUND),
    (r'\xi',          'Funding depletion rate',                  1.0,     rel(1.0, REL_FUND),     BASIS_FUND),
]

symbols = [p[0] for p in PARAMS]
param_latex = [f"${s}$" for s in symbols]
baseline = np.array([p[2] for p in PARAMS], dtype=float)
lower = np.array([p[3][0] for p in PARAMS], dtype=float)
upper = np.array([p[3][1] for p in PARAMS], dtype=float)
n_params = len(PARAMS)
assert np.all(lower < upper) and np.all((baseline >= lower) & (baseline <= upper)), \
    "baseline must lie inside its range"

IDX_NU = symbols.index(r'\nu')
IDX_PHI = symbols.index(r'\phi')


def tex_num(x):
    """Format a number for LaTeX (scientific notation where needed)."""
    if x != 0 and (abs(x) < 1e-2 or abs(x) >= 1e5):
        m, e = f"{x:.3e}".split("e")
        m = m.rstrip("0").rstrip(".")
        return rf"${m}\times 10^{{{int(e)}}}$"
    return f"${x:.4g}$"


def write_parameter_range_table(path="parameter_ranges.tex"):
    rows = []
    for s, desc, b, (lo, hi), basis in PARAMS:
        rows.append(
            f"${s}$ & {desc} & {tex_num(b)} & {tex_num(lo)} & {tex_num(hi)} & {DIST} & {basis} \\\\"
        )
    tex = (
        "\\begin{table}[H]\n\\centering\\small\n"
        "\\caption{Baseline values, ranges of variation and sampling distributions of the model "
        "parameters used in the sensitivity analysis (LHS--PRCC).}\n"
        "\\label{tab:param_ranges}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{llccccl}\n\\hline\n"
        "Parameter & Description & Baseline & Lower & Upper & Distribution & Basis of range \\\\\n\\hline\n"
        + "\n".join(rows)
        + "\n\\hline\n\\end{tabular}}\n\\end{table}\n"
    )
    with open(path, "w") as f:
        f.write(tex)

    pd.DataFrame({
        "Parameter": symbols,
        "Description": [p[1] for p in PARAMS],
        "Baseline": baseline,
        "Lower": lower,
        "Upper": upper,
        "Distribution": DIST,
    }).to_csv("parameter_ranges.csv", index=False)
    print(f"Saved '{path}' and 'parameter_ranges.csv'")


# ------------------------------------------------------------
# 2. Model RHS with funding saturation
# ------------------------------------------------------------
def abc_hpv_rhs_funding(t, S, V, Ia, As, Is, Ic, R, F, params):
    (beta, mu, alpha, sigma, delta, phi, gamma_, vareps,
     rho, omega, theta, tau, k1, k2, k3, Pi, nu,
     beta1, beta2, Lambda_F, eta1, eta2, xi) = params

    def sat(k, F):
        return k * (1.0 + beta1 * F / (1.0 + beta2 * F))

    omega_eff = sat(omega, F)
    tau_eff = min(sat(tau, F), 1.0)          # proportion cannot exceed 1
    sigma_eff = sat(sigma, F)
    delta_eff = sat(delta, F)
    vareps_eff = sat(vareps, F)

    N = S + V + Ia + As + Is + Ic + R
    if N <= 0:
        N = 1e-10

    lam = beta * (Ia + k1 * As + k2 * Is + k3 * Ic) / N

    dS = Pi - (lam + mu + omega_eff) * S + rho * R + theta * V
    dV = omega_eff * S - (theta + mu) * V
    dIa = lam * S - (mu + alpha) * Ia
    dAs = tau_eff * alpha * Ia - (sigma_eff + mu) * As
    dIs = (1 - tau_eff) * alpha * Ia - (mu + delta_eff + phi) * Is
    dIc = phi * Is - (mu + gamma_ + vareps_eff) * Ic
    dR = sigma_eff * As + delta_eff * Is + vareps_eff * Ic - (mu + rho) * R
    dF = Lambda_F + eta1 * Is + eta2 * Ic - xi * F

    return np.array([dS, dV, dIa, dAs, dIs, dIc, dR, dF])


# ------------------------------------------------------------
# 3. ABM weights (vectorised)
# ------------------------------------------------------------
def precompute_weights(nu, h, N_steps):
    Bnu = 1 - nu + nu / gamma(nu)
    factor = nu * (h ** nu) / (Bnu * gamma(nu + 2))

    n = np.arange(N_steps)[:, None]
    k = np.arange(N_steps)[None, :]
    mask = k <= n
    n1k = np.maximum(n + 1 - k, 0).astype(float)
    nk = np.maximum(n - k, 0).astype(float)

    A = (n1k ** nu) * (n1k + 1 + nu) - (nk ** nu) * (nk + 2 + 2 * nu)
    A = np.where(mask, A, 0.0)
    B = (n1k ** (nu + 1)) - (nk ** nu) * (nk + 1 + nu)
    B = np.where(mask & (k >= 1), B, 0.0)
    return A, B, Bnu, factor


# ------------------------------------------------------------
# 4. ABC Adams-Bashforth-Moulton solver (nu passed explicitly)
# ------------------------------------------------------------
def abc_abm_solver(F_rhs, tspan, y0, h, params, nu, weights):
    t0, T = tspan
    N_steps = int(round((T - t0) / h)) + 1
    t = np.linspace(t0, T, N_steps)
    y = np.zeros((N_steps, 8))
    y[0] = y0
    A, B, Bnu, factor = weights
    F_vals = np.zeros((N_steps, 8))
    F_vals[0] = F_rhs(t[0], *y[0], params)

    for n in range(N_steps - 1):
        sumA = A[n, :n + 1] @ F_vals[:n + 1]
        sumB = B[n, 1:n + 1] @ F_vals[:n] if n >= 1 else 0.0
        Fn = F_vals[n]
        y_pred = y0 + (1 - nu) / Bnu * Fn + factor * (sumA - sumB)
        F_pred = F_rhs(t[n + 1], *y_pred, params)
        y_corr = y0 + (1 - nu) / Bnu * F_pred + factor * (sumA - sumB + F_pred)
        y[n + 1] = y_corr
        F_vals[n + 1] = F_rhs(t[n + 1], *y_corr, params)
    return t, y


def cumulative_cancer(params, y0, tspan, h):
    nu = params[IDX_NU]
    N_steps = int(round((tspan[1] - tspan[0]) / h)) + 1
    weights = precompute_weights(nu, h, N_steps)
    t, y = abc_abm_solver(abc_hpv_rhs_funding, tspan, np.asarray(y0, float), h, params, nu, weights)
    if not np.all(np.isfinite(y)):
        return np.nan
    return trapezoid(params[IDX_PHI] * y[:, 5], t)


# ------------------------------------------------------------
# 5. Latin Hypercube Sampling
# ------------------------------------------------------------
N_SAMPLES = int(os.environ.get("N_SAMPLES", 1000))   # >= 500 recommended for 23 parameters
SEED = 42

sampler = qmc.LatinHypercube(d=n_params, seed=SEED)
lhs = qmc.scale(sampler.random(N_SAMPLES), lower, upper)

y0 = [600000, 350000, 250000, 240000, 220000, 200000, 0, 18.9]
tspan = (0, 30)
h = 0.1

write_parameter_range_table()

outputs = np.full(N_SAMPLES, np.nan)
print(f"Running {N_SAMPLES} LHS simulations (h={h}) ...")
for i in range(N_SAMPLES):
    try:
        outputs[i] = cumulative_cancer(lhs[i], y0, tspan, h)
    except Exception as e:
        print(f"Sample {i} failed: {e}")

valid = np.isfinite(outputs)
lhs_valid, outputs_valid = lhs[valid], outputs[valid]
print(f"Valid samples: {valid.sum()} out of {N_SAMPLES}")


# ------------------------------------------------------------
# 6. PRCC (with intercept in the partialling regressions)
# ------------------------------------------------------------
def prcc(x, y):
    x_rank = np.apply_along_axis(stats.rankdata, 0, x)
    y_rank = stats.rankdata(y)
    n, p = x_rank.shape
    vals, pvals = [], []
    for i in range(p):
        Z = np.column_stack([np.ones(n), np.delete(x_rank, i, axis=1)])
        rx = x_rank[:, i] - Z @ np.linalg.lstsq(Z, x_rank[:, i], rcond=None)[0]
        ry = y_rank - Z @ np.linalg.lstsq(Z, y_rank, rcond=None)[0]
        r, _ = stats.pearsonr(rx, ry)
        # p-value with n - 2 - (p - 1) degrees of freedom
        dof = n - 2 - (p - 1)
        tstat = r * np.sqrt(dof / (1 - r ** 2))
        vals.append(r)
        pvals.append(2 * stats.t.sf(abs(tstat), dof))
    return np.array(vals), np.array(pvals)


prcc_vals, p_vals = prcc(lhs_valid, outputs_valid)


def significance(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''


results_df = pd.DataFrame({
    'Parameter': param_latex,
    'Lower': lower,
    'Upper': upper,
    'PRCC': prcc_vals,
    'p-value': p_vals,
})
results_df['Significance'] = results_df['p-value'].apply(significance)
results_df = results_df.sort_values(by='PRCC', key=np.abs, ascending=False)
results_df.to_csv("sensitivity_results_funding.csv", index=False)
print("CSV saved as 'sensitivity_results_funding.csv'")

# ------------------------------------------------------------
# 7. PRCC LaTeX table (now shows the range used for each parameter)
# ------------------------------------------------------------
latex_table = (
    "\\begin{table}[H]\n\\centering\n"
    "\\caption{PRCC sensitivity analysis for cumulative HPV cancer cases with the funding model "
    f"($N={int(valid.sum())}$ LHS samples; ranges as in Table~\\ref{{tab:param_ranges}}).}}\n"
    "\\label{tab:prcc}\n\\begin{tabular}{lccccc}\n\\hline\n"
    "Parameter & Lower & Upper & PRCC & $p$-value & Significance \\\\\n\\hline\n"
)
def p_tex(p):
    return "$<0.0001$" if p < 1e-4 else f"{p:.4f}"

for _, r in results_df.iterrows():
    latex_table += (f"{r['Parameter']} & {tex_num(r['Lower'])} & {tex_num(r['Upper'])} & "
                    f"${r['PRCC']:.4f}$ & {p_tex(r['p-value'])} & {r['Significance']} \\\\\n")
latex_table += "\\hline\n\\end{tabular}\n\\end{table}\n"
with open("prcc_table_funding.tex", "w") as f:
    f.write(latex_table)
print("LaTeX table saved as 'prcc_table_funding.tex'")

# ------------------------------------------------------------
# 8. Bar chart
# ------------------------------------------------------------
os.makedirs('ImagesF', exist_ok=True)
plt.figure(figsize=(12, 8))
colors = ['red' if v < 0 else 'blue' for v in results_df['PRCC']]
plt.barh(results_df['Parameter'], results_df['PRCC'], color=colors)
plt.axvline(0, color='black', linewidth=0.8)
plt.xlabel('PRCC')
plt.title('Sensitivity Analysis – Funding Model')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("ImagesF/prcc_barchart_funding.png", dpi=300, bbox_inches='tight')
plt.close()
print("Bar chart saved as 'ImagesF/prcc_barchart_funding.png'")
print("Sensitivity analysis completed successfully.")