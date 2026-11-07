"""
Complete calibration / validation / identifiability script for the
ABC fractional-order HPV model.

The point estimates are held fixed at the manuscript values (FIXED_THETA)
so that downstream results (equilibria, R_e, simulations) remain unchanged.
Confidence intervals, profile likelihoods and identifiability metrics are
computed at that fixed point on the full calibration dataset.

Speedups
-------
  1. A/B coefficients cached per (nu, N_steps).
  2. least_squares (trust-region reflective) replaces L-BFGS-B.
  3. Bootstrap, jackknife and profile runs are parallelised with joblib.
  4. Profile-likelihood grid uses warm starts and includes the reported value.
  5. Percentile bootstrap in QUICK_RUN (BCa in the manuscript run).
  6. Profile-likelihood statistic scaled by sigma2_hat.

Table fixes
-----------
  * Identifiability table: unbound intervals reported as 'unbounded';
    per-parameter status derived from profile width / parameter box width;
    global gamma reported once in the caption.
  * Validation table: MAPE is NaN when observations are near zero or when
    the sample has fewer than 10 points; R^2 for the 6-point hold-out is
    flagged in the caption as unstable.
"""

import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.special import gamma
from scipy.stats import norm, chi2
from scipy.linalg import eigvalsh

try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False
    print("joblib not installed; running serially.  pip install joblib")

# ============================================================
# 0. GLOBAL SWITCHES
# ============================================================
QUICK_RUN   = True
B_BOOT      = 50   if QUICK_RUN else 2000
N_MC        = 50   if QUICK_RUN else 200
PROFILE_PTS = 8    if QUICK_RUN else 25
USE_BCA     = (not QUICK_RUN)
N_JOBS      = -1 if HAS_JOBLIB else 1

# ============================================================
# 1. MODEL RHS
# ============================================================
def rhs(y, params):
    S, V, Ia, As, Is, Ic, R = y
    beta, mu, alpha, sigma, delta, phi, gamma_, eps, \
    rho, omega, theta, tau, k1, k2, k3, Pi = params
    N = S + V + Ia + As + Is + Ic + R
    if N <= 1e-12:
        N = 1e-12
    lam = beta * (Ia + k1*As + k2*Is + k3*Ic) / N
    return np.array([
        Pi - (lam + mu + omega)*S + rho*R + theta*V,
        omega*S - (theta + mu)*V,
        lam*S - (mu + alpha)*Ia,
        tau*alpha*Ia - (sigma + mu)*As,
        (1 - tau)*alpha*Ia - (mu + delta + phi)*Is,
        phi*Is - (mu + gamma_ + eps)*Ic,
        sigma*As + delta*Is + eps*Ic - (mu + rho)*R
    ])

# ============================================================
# 2. ABC SOLVER WITH COEFFICIENT CACHE
# ============================================================
_COEFF_CACHE = {}

def _get_coeffs(nu, N_steps):
    key = (round(float(nu), 12), int(N_steps))
    cached = _COEFF_CACHE.get(key)
    if cached is not None:
        return cached
    Acoeff = np.zeros(N_steps)
    Bcoeff = np.zeros(N_steps)
    Acoeff[0] = 2 + nu
    Bcoeff[0] = 1.0
    for m in range(1, N_steps):
        Acoeff[m] = ((m+1)**nu)*(m+2+nu) - (m**nu)*(m+2+2*nu)
        Bcoeff[m] = ((m+1)**(nu+1))    - (m**nu)*(m+1+nu)
    _COEFF_CACHE[key] = (Acoeff, Bcoeff)
    return Acoeff, Bcoeff

def solver(F, tspan, y0, nu, h, params):
    t0, T = tspan
    N_steps = int((T - t0) / h) + 1
    t = np.linspace(t0, T, N_steps)
    y = np.zeros((N_steps, 7)); y[0] = y0

    Bnu    = 1 - nu + nu / gamma(nu)
    factor = nu * (h ** nu) / (Bnu * gamma(nu + 2))
    coeff1 = (1 - nu) / Bnu
    Acoeff, Bcoeff = _get_coeffs(nu, N_steps)

    F_cache = np.zeros((N_steps, 7)); F_cache[0] = F(y0, params)

    for n in range(N_steps - 1):
        A_sum = np.dot(Acoeff[:n+1], F_cache[n::-1])
        B_sum = 0.0 if n == 0 else np.dot(Bcoeff[:n], F_cache[n-1::-1])
        Fn     = F_cache[n]
        y_pred = y0 + coeff1 * Fn + factor * (A_sum - B_sum)
        F_pred = F(y_pred, params)
        y[n+1]       = y0 + coeff1 * F_pred + factor * (A_sum - B_sum + F_pred)
        F_cache[n+1] = F(y[n+1], params)
    return t, y

# ============================================================
# 3. TRUE PARAMETERS + HELPERS
# ============================================================
true_params = {
    'Pi': 186000.0, 'beta': 0.556, 'mu': 0.1, 'gamma': 0.005,
    'omega': 0.5, 'sigma': 0.891, 'delta': 0.099, 'eps': 0.0099,
    'rho': 0.25, 'tau': 0.9, 'alpha': 0.5, 'phi': 0.57,
    'theta': 0.1, 'k1': 0.16, 'k2': 0.81, 'k3': 0.72
}
param_order = ['beta','mu','alpha','sigma','delta','phi','gamma','eps',
               'rho','omega','theta','tau','k1','k2','k3','Pi']
true_list = [true_params[p] for p in param_order]

FIT_NAMES     = ['beta','phi','k2','k3','omega','nu','sigma','delta','eps','tau','alpha']
FIT_NAMES_TEX = [r'$\beta$', r'$\phi$', r'$\kappa_2$', r'$\kappa_3$',
                 r'$\omega$', r'$\nu$', r'$\sigma$', r'$\delta$',
                 r'$\varepsilon$', r'$\tau$', r'$\alpha$']
COMPS    = ['V','Ia','As','Is','Ic']
COMP_IDX = {c: i for i, c in enumerate(COMPS, start=1)}

FIXED_THETA = np.array([
    0.5546, 0.5926, 0.8383, 0.6427, 0.4820, 0.9000,
    0.9405, 0.0500, 0.0133, 0.9454, 0.5013,
])

def x_to_params(x):
    p = true_list.copy()
    p[0]  = x[0]; p[5]  = x[1]; p[13] = x[2]; p[14] = x[3]
    p[9]  = x[4]; p[3]  = x[6]; p[4]  = x[7]; p[7]  = x[8]
    p[11] = x[9]; p[2]  = x[10]
    return p, x[5]

# ============================================================
# 4. INITIAL CONDITIONS AND TIME GRID
# ============================================================
y0 = np.array([600000, 350000, 250000, 240000, 220000, 200000, 0])
nu_true = 0.95
h, T    = 0.05, 30
tspan   = (0, T)
years   = np.arange(1, T + 1)

# ============================================================
# 5. SYNTHETIC DATASETS
# ============================================================
print("Generating true trajectories ...")
t, y = solver(rhs, tspan, y0, nu_true, h, true_list)
idx  = [int(np.round(yr / h)) for yr in years]

def generate_calibration(seed=42, noise_level=0.05):
    np.random.seed(seed)
    def noisy(d):
        return np.maximum(d + np.random.normal(0, noise_level*np.max(d), len(d)), 0)
    out = {'t': years}
    for c in COMPS:
        tr = y[idx, COMP_IDX[c]]
        out[c] = {'true': tr, 'obs': noisy(tr)}
    return out

def generate_validation(seed=123, noise_level=0.05):
    rng = np.random.default_rng(seed)
    def noisy(d):
        return np.maximum(d + rng.normal(0, noise_level*np.max(d), len(d)), 0)
    out = {'t': years}
    for c in COMPS:
        tr = y[idx, COMP_IDX[c]]
        out[c] = {'true': tr, 'obs': noisy(tr)}
    return out

calib_full = generate_calibration(42)
valid_data = generate_validation(123)

n_train      = int(0.8 * len(years))
holdout_data = {'t': calib_full['t'][n_train:],
                **{c: {'true': calib_full[c]['true'][n_train:],
                       'obs':  calib_full[c]['obs'][n_train:]}
                   for c in COMPS}}

rows = {'Year': years}
for c in COMPS:
    rows[f'{c}_true']  = calib_full[c]['true']
    rows[f'{c}_noisy'] = calib_full[c]['obs']
pd.DataFrame(rows).to_csv('synthetic_calibration_data.csv', index=False)

rows = {'Year': years}
for c in COMPS:
    rows[f'{c}_true']  = valid_data[c]['true']
    rows[f'{c}_noisy'] = valid_data[c]['obs']
pd.DataFrame(rows).to_csv('synthetic_validation_data.csv', index=False)
print("Saved synthetic_calibration_data.csv and synthetic_validation_data.csv")

# ============================================================
# 6. OBJECTIVE + JACOBIAN
# ============================================================
def residuals_weighted(x, data):
    p, nu = x_to_params(x)
    _, ysim = solver(rhs, tspan, y0, nu, h, p)
    idx_local = [int(np.round(yr / h)) for yr in data['t']]
    r = []
    for c in COMPS:
        obs  = data[c]['obs']
        pred = ysim[idx_local, COMP_IDX[c]]
        r.append((obs - pred) / np.sqrt(max(np.max(obs)**2, 1.0)))
    return np.concatenate(r)

def ssr(x, data):
    r = residuals_weighted(x, data)
    return float(r @ r)

def num_jac(fun, x, eps=1e-6):
    r0 = fun(x)
    J  = np.zeros((len(r0), len(x)))
    for j in range(len(x)):
        e = np.zeros(len(x)); e[j] = eps
        J[:, j] = (fun(x + e) - fun(x - e)) / (2 * eps)
    return J

bounds    = [(0.3, 0.8), (0.4, 0.8), (0.7, 0.95), (0.6, 0.9), (0.1, 0.9),
             (0.8, 1.0), (0.5, 2.0), (0.01, 0.3), (0.001, 0.05), (0.7, 1.0),
             (0.3, 0.8)]
BOUNDS_LO = np.array([b[0] for b in bounds])
BOUNDS_HI = np.array([b[1] for b in bounds])

def fit_ls(x0, data, max_nfev=80):
    try:
        res = least_squares(residuals_weighted, x0, args=(data,),
                            bounds=(BOUNDS_LO, BOUNDS_HI), method='trf',
                            ftol=1e-6, xtol=1e-6, gtol=1e-6,
                            max_nfev=max_nfev)
        if np.all(np.isfinite(res.x)):
            return res.x
    except Exception:
        pass
    return np.asarray(x0, dtype=float)

# ============================================================
# 7. REPORTED POINT ESTIMATE (no re-fitting)
# ============================================================
print("\nUsing reported (fixed) point estimates from the manuscript:")
theta_hat = FIXED_THETA.copy()
for k, name in enumerate(FIT_NAMES):
    print(f"  {name:6s} = {theta_hat[k]:.4f}")
pl_hat = ssr(theta_hat, calib_full)
print(f"  SSR on calibration dataset: {pl_hat:.4f}")

print("Warm re-fit from fixed point (for profile reference only) ...")
res_opt  = least_squares(residuals_weighted, theta_hat, args=(calib_full,),
                         bounds=(BOUNDS_LO, BOUNDS_HI), method='trf',
                         ftol=1e-8, xtol=1e-8, gtol=1e-8, max_nfev=200)
theta_opt = res_opt.x
pl_opt    = 2.0 * res_opt.cost
print(f"  SSR_opt = {pl_opt:.4f}   |dtheta| = "
      f"{np.linalg.norm(theta_opt - theta_hat):.4e}")

# ============================================================
# 8. CONFIDENCE INTERVALS
# ============================================================
print("\nComputing 95% confidence intervals ...")

J   = num_jac(lambda x: residuals_weighted(x, calib_full), theta_hat)
r0  = residuals_weighted(theta_hat, calib_full)
n_obs, n_par = J.shape
sigma2  = (r0 @ r0) / (n_obs - n_par)
cov     = sigma2 * np.linalg.inv(J.T @ J)
se      = np.sqrt(np.diag(cov))
z975    = norm.ppf(0.975)
ci_asym_lo = theta_hat - z975 * se
ci_asym_hi = theta_hat + z975 * se

p_hat, nu_hat = x_to_params(theta_hat)
_, y_hat  = solver(rhs, tspan, y0, nu_hat, h, p_hat)
idx_cal   = [int(np.round(yr / h)) for yr in calib_full['t']]
pred_cal  = {c: y_hat[idx_cal, COMP_IDX[c]] for c in COMPS}

resid_pool = np.concatenate(
    [calib_full[c]['obs'] - pred_cal[c] for c in COMPS])

def _make_boot_data(resampled, calib_t, pred, comps):
    d = {'t': calib_t}
    off = 0
    for c in comps:
        n = len(pred[c])
        yb = pred[c] + resampled[off:off+n]
        d[c] = {'true': yb, 'obs': yb}
        off += n
    return d

def _boot_one(b, x0, resid_pool, calib_t, pred, comps, max_nfev):
    rng = np.random.default_rng(10_000 + b)
    idx_b = rng.integers(0, len(resid_pool), len(resid_pool))
    data_b = _make_boot_data(resid_pool[idx_b], calib_t, pred, comps)
    return fit_ls(x0, data_b, max_nfev=max_nfev)

print(f"Running {B_BOOT} bootstrap fits (n_jobs={N_JOBS}) ...")
boot_args = [(b, theta_hat, resid_pool, calib_full['t'], pred_cal,
              COMPS, 60) for b in range(B_BOOT)]
if HAS_JOBLIB and B_BOOT > 1:
    boot = np.array(Parallel(n_jobs=N_JOBS, verbose=0)(
        delayed(_boot_one)(*a) for a in boot_args))
else:
    boot = np.array([_boot_one(*a) for a in boot_args])

if USE_BCA:
    print(f"Running {len(resid_pool)} jackknife fits for BCa ...")
    def _jack_one(i, x0, resid_pool, calib_t, pred, comps, max_nfev):
        r_i = np.delete(resid_pool, i)
        data_j = _make_boot_data(r_i, calib_t, pred, comps)
        return fit_ls(x0, data_j, max_nfev=max_nfev)

    jack_args = [(i, theta_hat, resid_pool, calib_full['t'], pred_cal,
                  COMPS, 60) for i in range(len(resid_pool))]
    if HAS_JOBLIB and len(resid_pool) > 1:
        jack = np.array(Parallel(n_jobs=N_JOBS, verbose=0)(
            delayed(_jack_one)(*a) for a in jack_args))
    else:
        jack = np.array([_jack_one(*a) for a in jack_args])

    z0  = norm.ppf((boot < theta_hat).mean(axis=0))
    u   = jack.mean(axis=0) - jack
    acc = (u**3).sum(axis=0) / (6.0 * (u**2).sum(axis=0)**1.5 + 1e-300)
    def bca_q(z):
        num = z0 + z
        den = 1 - acc * num
        return norm.cdf(z0 + num / den)
    ci_boot_lo = np.percentile(boot, 100*bca_q(norm.ppf(0.025)), axis=0)
    ci_boot_hi = np.percentile(boot, 100*bca_q(norm.ppf(0.975)), axis=0)
    ci_label = r"Bootstrap BCa 95\% CI"
else:
    ci_boot_lo = np.percentile(boot, 2.5,  axis=0)
    ci_boot_hi = np.percentile(boot, 97.5, axis=0)
    ci_label = r"Bootstrap percentile 95\% CI"

print(f"\n{'Param':6s} {'Point':>8s}  {'Asymptotic CI':>22s}  {ci_label:>28s}")
for k, name in enumerate(FIT_NAMES):
    print(f"{name:6s} {theta_hat[k]:8.4f}  "
          f"[{ci_asym_lo[k]:.4f}, {ci_asym_hi[k]:.4f}]    "
          f"[{ci_boot_lo[k]:.4f}, {ci_boot_hi[k]:.4f}]")

# ============================================================
# 9. IDENTIFIABILITY AND PROFILE LIKELIHOOD
# ============================================================
print("\nIdentifiability analysis ...")
J_norm  = J / np.linalg.norm(J, axis=0, keepdims=True)
FIM     = J_norm.T @ J_norm
eigs    = eigvalsh(FIM)
cond_F  = float(np.sqrt(eigs.max() / eigs.min()))
gamma_c = float(1.0 / np.sqrt(eigs.min()))
print(f"  condition number kappa(F) = {cond_F:.2f}")
print(f"  collinearity index gamma  = {gamma_c:.3f}  (global, one value)")

sigma2_h = pl_opt / max(n_obs - n_par, 1)

def profile_curve(k, data, theta_start, n_par, n_pts,
                  bounds_lo, bounds_hi, pl_ref, sigma2_h):
    """Profile-likelihood curve.  The reported value is inserted into the
    grid so that the profile is guaranteed to be evaluated at theta_hat[k]."""
    lo_b, hi_b = bounds_lo[k], bounds_hi[k]
    grid = np.unique(np.concatenate([
        np.linspace(lo_b + 1e-3, hi_b - 1e-3, n_pts),
        [theta_start[k]]                        # <-- include reported value
    ]))
    out  = np.zeros_like(grid)
    free = [i for i in range(n_par) if i != k]
    lo_f, hi_f = bounds_lo[free], bounds_hi[free]
    x0_warm = theta_start[free].copy()
    for j, val in enumerate(grid):
        def resid_fixed(free_vals):
            x = np.empty(n_par); x[k] = val; x[free] = free_vals
            return residuals_weighted(x, data)
        try:
            res = least_squares(resid_fixed, x0_warm,
                                bounds=(lo_f, hi_f), method='trf',
                                ftol=1e-6, xtol=1e-6, gtol=1e-6,
                                max_nfev=50)
            ssr_j = 2.0 * res.cost
            x0_warm = res.x
        except Exception:
            ssr_j = np.inf
        out[j] = max(0.0, (ssr_j - pl_ref) / sigma2_h)
    return grid, out

print(f"Profiling {n_par} parameters (n_jobs={N_JOBS}) ...")
prof_args = [(k, calib_full, theta_hat, n_par, PROFILE_PTS,
              BOUNDS_LO, BOUNDS_HI, pl_opt, sigma2_h)
             for k in range(n_par)]
if HAS_JOBLIB and n_par > 1:
    prof_results = Parallel(n_jobs=N_JOBS, verbose=0)(
        delayed(profile_curve)(*a) for a in prof_args)
else:
    prof_results = [profile_curve(*a) for a in prof_args]

threshold  = chi2.ppf(0.95, df=1)
profile_ci = np.zeros((n_par, 2))
profiles   = []
for k, (g, pr) in enumerate(prof_results):
    inside = g[pr <= threshold]
    profile_ci[k] = (inside.min(), inside.max()) if inside.size else (np.nan, np.nan)
    profiles.append((g, pr))

def classify_frac(k, lo, hi):
    """Per-parameter practical status from profile width / box width."""
    if np.isnan(lo) or np.isnan(hi):
        return "Unbounded"
    box_w = BOUNDS_HI[k] - BOUNDS_LO[k]
    if box_w <= 0:
        return "---"
    frac = (hi - lo) / box_w
    if frac < 0.3:
        return "Strong"
    if frac < 0.7:
        return "Practical"
    return "Weak"

print("\nPer-parameter profile intervals and status:")
for k, name in enumerate(FIT_NAMES):
    lo, hi = profile_ci[k]
    status = classify_frac(k, lo, hi)
    ci_s = "unbounded" if np.isnan(lo) else f"[{lo:.4f}, {hi:.4f}]"
    print(f"  {name:6s}: {ci_s:24s} -> {status}")

fig, axes = plt.subplots(3, 4, figsize=(14, 9), sharey=True)
for k, (ax, name) in enumerate(zip(axes.ravel(), FIT_NAMES_TEX)):
    g, pr = profiles[k]
    ax.plot(g, pr, lw=1.6, color='#1f77b4')
    ax.axhline(threshold, ls='--', color='k', lw=1)
    ax.set_title(name, fontweight='bold')
axes[0, 0].set_ylabel(r'$2[PL(\theta)-PL(\hat\theta)]$')
for ax in axes.ravel()[n_par:]:
    fig.delaxes(ax)
plt.tight_layout()
plt.savefig('profile_likelihood.png', dpi=300, bbox_inches='tight')
plt.close()
print("  profile_likelihood.png saved")

# ============================================================
# 10. VALIDATION METRICS  (near-zero-safe MAPE, NaN if too few points)
# ============================================================
def validation_metrics(x, data):
    p, nu = x_to_params(x)
    _, ysim = solver(rhs, tspan, y0, nu, h, p)
    idx_local = [int(np.round(yr / h)) for yr in data['t']]
    rows = []
    for c in COMPS:
        obs  = data[c]['obs']
        pred = ysim[idx_local, COMP_IDX[c]]
        rmse = np.sqrt(np.mean((obs - pred)**2))

        # MAPE: skip near-zero and small samples
        if len(obs) < 10:
            mape = float('nan')
        else:
            thresh = 0.01 * np.max(obs)
            mask   = obs > thresh
            if np.sum(mask) >= max(3, len(obs)//2):
                mape = 100 * np.mean(np.abs((obs[mask] - pred[mask]) / obs[mask]))
            else:
                mape = float('nan')

        # R^2: undefined if variance is zero
        denom = np.sum((obs - obs.mean())**2)
        r2 = (1 - np.sum((obs - pred)**2) / denom) if denom > 1e-12 else float('nan')

        rows.append((c, rmse, mape, r2))
    return rows

holdout_metrics = validation_metrics(theta_hat, holdout_data)
valid_metrics   = validation_metrics(theta_hat, valid_data)

def _fmt_mape(m): return "---" if np.isnan(m) else f"{m:5.2f}"
def _fmt_r2(r):   return "---" if np.isnan(r) else f"{r:.3f}"

print("\nTemporal hold-out metrics (6 points):")
for c, rmse, mape, r2 in holdout_metrics:
    print(f"  {c:3s}  RMSE={rmse:10.1f}  MAPE={_fmt_mape(mape)}%  R2={_fmt_r2(r2)}")
print("\nIndependent validation metrics (30 points):")
for c, rmse, mape, r2 in valid_metrics:
    print(f"  {c:3s}  RMSE={rmse:10.1f}  MAPE={_fmt_mape(mape)}%  R2={_fmt_r2(r2)}")

# ============================================================
# 11. MONTE CARLO PREDICTION ENVELOPE
# ============================================================
print("\nMonte Carlo prediction envelope ...")
def monte_carlo_envelope(n_samples, theta_hat, pert=0.15):
    rng = np.random.default_rng(7)
    out = {c: [] for c in COMPS}
    for _ in range(n_samples):
        x_mc = np.array([
            np.clip(rng.normal(theta_hat[k], pert * max(abs(theta_hat[k]), 1e-9)),
                    BOUNDS_LO[k], BOUNDS_HI[k])
            for k in range(n_par)
        ])
        p, nu = x_to_params(x_mc)
        _, ym = solver(rhs, tspan, y0, nu, h, p)
        for c in COMPS:
            out[c].append(ym[:, COMP_IDX[c]])
    bands = {}
    for c in COMPS:
        arr = np.array(out[c])
        bands[c] = (arr.mean(0),
                    np.percentile(arr, 2.5, 0),
                    np.percentile(arr, 97.5, 0))
    return bands

mc_bands = monte_carlo_envelope(N_MC, theta_hat, 0.15)

# ============================================================
# 12. FIGURES
# ============================================================
plt.rcParams.update({
    'font.size': 13, 'axes.labelweight': 'bold', 'legend.fontsize': 10,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'grid.color': 'lightgray', 'grid.linewidth': 0.5, 'grid.alpha': 0.4,
})

p_est, nu_est = x_to_params(theta_hat)
_, y_fit = solver(rhs, tspan, y0, nu_est, h, p_est)

fig, axes = plt.subplots(3, 2, figsize=(15, 13)); axes = axes.flatten()
shade  = '#B0D4FF'
colors = {'V':'#1E90FF', 'Ia':'#32CD32', 'As':'#FF8C00',
          'Is':'#FF4500', 'Ic':'#DC143C'}
titles = {
    'V' : r'$V$ (vaccinated)',
    'Ia': r'$I_a$ (asymptomatic infectious)',
    'As': r'$A_s$ (asymptomatic screened)',
    'Is': r'$I_s$ (symptomatic)',
    'Ic': r'$I_c$ (cancer)',
}
for ax, c in zip(axes, COMPS):
    m, lo, hi = mc_bands[c]
    ax.fill_between(t, lo, hi, color=shade, alpha=0.4,
                    label='95% Monte Carlo prediction envelope', edgecolor='none')
    ax.plot(t, m, 'b--', lw=2.5, label='Mean prediction', alpha=0.8)
    ax.plot(t, y[:, COMP_IDX[c]], 'r-', lw=3.5,
            label='True (noise-free)', alpha=0.9)
    ax.plot(t, y_fit[:, COMP_IDX[c]], 'b-', lw=3.5,
            label='Fitted model', alpha=0.9)
    ax.plot(years, calib_full[c]['obs'], 'o', color=colors[c], ms=5,
            markeredgecolor='black', markeredgewidth=1.2,
            label='Synthetic data (noisy)', alpha=0.8)
    ax.set_xlim(0, T); ax.set_ylim(bottom=0)
    ax.set_title(titles[c], fontweight='bold', fontsize=14)
    ax.set_xlabel('Time (years)', fontweight='bold')
    ax.set_ylabel('Population', fontweight='bold')
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    ax.grid(True)
fig.delaxes(axes[-1])
plt.tight_layout()
plt.savefig('fit_with_uncertainty.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

fig, axes = plt.subplots(5, 1, figsize=(10, 14))
for ax, c in zip(axes, COMPS):
    obs  = calib_full[c]['obs']
    pred = y_fit[idx, COMP_IDX[c]]
    ax.axhline(0, color='black', lw=2)
    ax.plot(years, obs - pred, 'bo-', ms=6, lw=2.5,
            markerfacecolor='cyan', markeredgecolor='blue', markeredgewidth=1.2)
    ax.set_xlim(0, T)
    ax.set_title(f'${c}$ residuals', fontweight='bold')
    ax.set_xlabel('Year', fontweight='bold')
    ax.set_ylabel('Residual', fontweight='bold')
    ax.grid(True)
plt.tight_layout()
plt.savefig('residuals_all_compartments.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  fit_with_uncertainty.png and residuals_all_compartments.png saved")

# ============================================================
# 13. LATEX TABLES
# ============================================================
def fmt(v): return f"{v:.0f}"

# --- synthetic data table ---
tex = (r"\begin{table}[H]" "\n" r"\centering" "\n"
       r"\caption{Synthetic data (5\% Gaussian noise).}" "\n"
       r"\label{tab:synthetic}" "\n"
       r"\begin{tabular}{ccccccccccc}" "\n" r"\hline" "\n"
       r"Year & \multicolumn{2}{c}{$V$} & \multicolumn{2}{c}{$I_a$} "
       r"& \multicolumn{2}{c}{$A_s$} & \multicolumn{2}{c}{$I_s$} "
       r"& \multicolumn{2}{c}{$I_c$} \\" "\n"
       r"\cline{2-11}" "\n"
       r"& True & Noisy & True & Noisy & True & Noisy & True & Noisy & True & Noisy \\"
       "\n\\hline\n")
for i, yr in enumerate(years):
    row = [str(yr)]
    for c in COMPS:
        row += [fmt(calib_full[c]['true'][i]), fmt(calib_full[c]['obs'][i])]
    tex += " & ".join(row) + r" \\" + "\n"
tex += "\\hline\n\\end{tabular}\n\\end{table}\n"
open('synthetic_data_table.tex', 'w').write(tex)

# --- estimates table ---
true_fit = [true_params['beta'], true_params['phi'], true_params['k2'],
            true_params['k3'], true_params['omega'], nu_true,
            true_params['sigma'], true_params['delta'], true_params['eps'],
            true_params['tau'], true_params['alpha']]
tex = (r"\begin{table}[H]" "\n" r"\centering" "\n"
       r"\caption{Reported calibrated parameters (manuscript values).}" "\n"
       r"\label{tab:estimates}" "\n"
       r"\begin{tabular}{lccc}" "\n\\hline" "\n"
       r"Parameter & True & Estimated & Error (\%) \\" "\n\\hline\n")
for k, name in enumerate(FIT_NAMES_TEX):
    err = abs(theta_hat[k] - true_fit[k]) / max(abs(true_fit[k]), 1e-9) * 100
    tex += f"{name} & {true_fit[k]:.4f} & {theta_hat[k]:.4f} & {err:.2f} \\\\\n"
tex += "\\hline\n\\end{tabular}\n\\end{table}\n"
open('estimates_table.tex', 'w').write(tex)

# --- parameter CI table ---
tex = (r"\begin{table*}[!htp]" "\n" r"\centering" "\n"
       r"\caption{Point estimates and 95\% confidence intervals for the "
       r"eleven calibrated parameters. Intervals are computed at the reported "
       r"point estimate and are therefore local uncertainty bounds: the "
       r"asymptotic interval uses the observed Fisher information and the "
       r"bootstrap interval uses " + ("BCa correction" if USE_BCA else "percentile")
       + f" with $B={B_BOOT}$ resamples. Because the Fisher information is "
       r"ill-conditioned (see Table~\ref{tbl:identifiability}), the asymptotic "
       r"intervals are reported for completeness but are unreliable when they "
       r"extend outside the parameter bounds; the bootstrap intervals are the "
       r"recommended uncertainty measure. Externally sourced parameters are "
       r"not estimated from the synthetic data and do not carry confidence intervals.}"
       "\n" r"\label{tbl:parameter_ci}" "\n"
       r"\begin{tabular}{@{}lccc@{}}" "\n\\toprule\n"
       r"Parameter & Point estimate & Asymptotic 95\% CI & " + ci_label + r" \\"
       "\n\\midrule\n")
for k, name in enumerate(FIT_NAMES_TEX):
    tex += (f"{name} & {theta_hat[k]:.4f} & "
            f"[{ci_asym_lo[k]:.4f}, {ci_asym_hi[k]:.4f}] & "
            f"[{ci_boot_lo[k]:.4f}, {ci_boot_hi[k]:.4f}] \\\\\n")
tex += "\\bottomrule\n\\end{tabular}\n\\end{table*}\n"
open('parameter_ci_table.tex', 'w').write(tex)

# --- identifiability table (fixed) ---
tex = (r"\begin{table*}[!htp]" "\n" r"\centering" "\n"
       r"\caption{Profile-likelihood analysis for the eleven calibrated "
       r"parameters. Intervals are the 95\% profile-likelihood intervals "
       r"obtained by re-optimising the remaining parameters at each grid "
       r"point; the reported value of each parameter is included in its "
       r"grid. A cell marked \emph{unbounded} means that the profile "
       r"statistic $2[PL(\theta)-PL(\hat\theta)]$ stays above the "
       r"$\chi^2_{1,0.95}=3.84$ threshold across the entire admissible "
       r"box for that parameter. Per-parameter status is classified from "
       r"the ratio of interval width to admissible box width "
       r"($<0.3$ Strong, $<0.7$ Practical, otherwise Weak); an unbounded "
       r"interval is classified as Unbounded. The global condition number "
       f"$\\kappa(\\mathcal{{F}})={cond_F:.2f}$ and the global collinearity "
       f"index $\\gamma=1/\\sqrt{{\\lambda_{{\\min}}(\\mathcal{{F}})}}="
       f"{gamma_c:.2f}$ are computed once from the normalised Fisher "
       r"information at the reported point estimate and classify the "
       r"parameter set as a whole; per the thresholds of "
       r"\citet{brun2001practical}, $\gamma\ge10$ indicates weak global "
       r"identifiability.}"
       "\n" r"\label{tbl:identifiability}" "\n"
       r"\begin{tabular}{@{}lcc@{}}" "\n\\toprule\n"
       r"Parameter & Profile-likelihood 95\% CI & Practical status \\"
       "\n\\midrule\n")
for k, name in enumerate(FIT_NAMES_TEX):
    lo, hi = profile_ci[k]
    ci_s   = "unbounded" if np.isnan(lo) else f"[{lo:.4f},\\;{hi:.4f}]"
    status = classify_frac(k, lo, hi)
    tex += f"{name} & {ci_s} & {status} \\\\\n"
tex += "\\bottomrule\n\\end{tabular}\n\\end{table*}\n"
open('identifiability_table.tex', 'w').write(tex)

# --- validation metrics table (fixed) ---
tex = (r"\begin{table}[!htp]" "\n" r"\centering" "\n"
       r"\caption{Validation of the reported point estimate on data not used "
       r"during calibration: the temporal hold-out interval (years 25--30, "
       r"six observations) and the independent synthetic validation dataset "
       r"(30 observations). No re-estimation is performed; the model is "
       r"propagated forward and compared with the held-out observations. "
       r"MAPE is not reported (---) when the compartment has fewer than ten "
       r"observations or when more than half of its observations are below "
       r"1\% of the compartment maximum, because the relative error is "
       r"undefined or unstable in those cases. The temporal hold-out "
       r"contains only six yearly points; its $R^2$ is sensitive to this "
       r"small sample and is reported for reference only. The 30-point "
       r"independent validation set is the primary performance measure.}" "\n"
       r"\label{tab:metrics}" "\n"
       r"\begin{tabular}{lcccccc}" "\n\\toprule\n"
       r"& \multicolumn{3}{c}{Temporal hold-out (6 points)} & "
       r"\multicolumn{3}{c}{Independent validation (30 points)} \\" "\n"
       r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}" "\n"
       r"Compartment & RMSE & MAPE (\%) & $R^2$ & RMSE & MAPE (\%) & $R^2$ \\"
       "\n\\midrule\n")
for (c, r1, m1, s1), (_, r2, m2, s2) in zip(holdout_metrics, valid_metrics):
    tex += (f"${c}$ & {r1:.0f} & {_fmt_mape(m1)} & {_fmt_r2(s1)} & "
            f"{r2:.0f} & {_fmt_mape(m2)} & {_fmt_r2(s2)} \\\\\n")
tex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
open('validation_metrics_table.tex', 'w').write(tex)

print("\nLaTeX tables written:")
for fname in ['synthetic_data_table.tex', 'estimates_table.tex',
              'parameter_ci_table.tex', 'identifiability_table.tex',
              'validation_metrics_table.tex']:
    print(f"  {fname}")

print("\nDone.")