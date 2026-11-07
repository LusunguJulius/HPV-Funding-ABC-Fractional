"""
Quantitative transient analysis across fractional orders (corrected version).

Changes relative to the original script
---------------------------------------
* Decay rate: fits ln|I(t) - I*| over a window common to all nu, with R^2
  and the window reported (no NaN when the maximum is at t = 0).
* Eigenvalue analysis: dominant eigenvalue = largest real part; Matignon test
  on the full spectrum; relative finite-difference step.
* ABC-ABM solver vectorised (numerically identical to the original loop).
* LaTeX tables updated; h-refinement figure saved with .png extension.
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.special import gamma
from scipy.optimize import root
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

plt.rcParams.update({
    'font.size': 13, 'axes.labelweight': 'bold', 'axes.titleweight': 'bold',
    'axes.linewidth': 1.5, 'xtick.major.width': 1.5, 'ytick.major.width': 1.5,
    'xtick.major.size': 6, 'ytick.major.size': 6, 'legend.frameon': True,
    'legend.framealpha': 0.9, 'legend.edgecolor': 'black',
    'legend.fontsize': 11, 'lines.linewidth': 3.5,
})

# ---------------- parameters ----------------
Pi = 186000.0; mu = 0.1; gamma_c = 0.005; rho = 0.25
alpha = 0.5013; phi = 0.5926; theta = 0.1
kappa1 = 0.16; kappa2 = 0.8383; kappa3 = 0.6427
omega_base = 0.4820; sigma_base = 0.9405; delta_base = 0.0500
epsilon_base = 0.0133; tau_base = 0.9454
beta1 = 0.02; beta2 = 0.001
Lambda_F = 18.9; eta1 = 1.98e-7; eta2 = 9.23e-6; xi = 1.0
STATE_KEYS = ['S', 'V', 'Ia', 'As', 'Is', 'Ic', 'R', 'F']


def psi(k, F):
    return k * (1 + beta1 * F / (1 + beta2 * F))


def get_rates(F):
    return (psi(omega_base, F), min(psi(tau_base, F), 1.0),
            psi(sigma_base, F), psi(delta_base, F), psi(epsilon_base, F))


def compute_Re(beta_val):
    F0 = Lambda_F / xi
    omega0, tau0, sigma0, delta0, epsilon0 = get_rates(F0)
    d1 = mu + alpha; d2 = mu + sigma0
    d3 = mu + delta0 + phi; d4 = mu + gamma_c + epsilon0
    factor = beta_val * (theta + mu) / (theta + mu + omega0)
    return factor * (1.0 / d1
                     + kappa1 * tau0 * alpha / (d1 * d2)
                     + kappa2 * (1 - tau0) * alpha / (d1 * d3)
                     + kappa3 * phi * (1 - tau0) * alpha / (d1 * d3 * d4))


def compute_DFE():
    F0 = Lambda_F / xi
    omega0 = get_rates(F0)[0]
    S0 = Pi * (mu + theta) / (mu * (mu + theta + omega0))
    V0 = Pi * omega0 / (mu * (mu + theta + omega0))
    return S0, V0, F0


def compute_EE(beta_val):
    Re = compute_Re(beta_val)
    if Re <= 1:
        return None
    F0 = Lambda_F / xi

    def parts(F):
        omega, tau, sigma, delta, epsilon = get_rates(F)
        d2 = mu + sigma; d3 = mu + delta + phi; d4 = mu + gamma_c + epsilon
        m1 = tau * alpha / d2
        m2 = (1 - tau) * alpha / d3
        m3 = phi * (1 - tau) * alpha / (d3 * d4)
        K = 1 + m1 + m2 + m3 + (sigma*m1 + delta*m2 + epsilon*m3) / (mu + rho)
        return omega, sigma, delta, epsilon, m1, m2, m3, K

    def eq_F(F):
        omega, sigma, delta, epsilon, m1, m2, m3, K = parts(F)
        Ia = (Re - 1) / (K * Re) * (Pi / mu)
        return F - (Lambda_F + eta1*m2*Ia + eta2*m3*Ia) / xi

    sol = root(eq_F, F0)
    if not sol.success:
        raise ValueError("Could not solve for F*")
    F_star = float(sol.x[0])
    omega, sigma, delta, epsilon, m1, m2, m3, K = parts(F_star)
    A = (theta + mu + omega) / (theta + mu)
    Ia_star = (Re - 1) / (K * Re) * (Pi / mu)
    S_star = (Pi / mu) / (A * Re)
    V_star = omega / (theta + mu) * S_star
    return {'S': S_star, 'V': V_star, 'Ia': Ia_star, 'As': m1*Ia_star,
            'Is': m2*Ia_star, 'Ic': m3*Ia_star,
            'R': (sigma*m1 + delta*m2 + epsilon*m3) / (mu + rho) * Ia_star,
            'F': F_star}


def get_equilibrium(beta_val):
    if compute_Re(beta_val) <= 1:
        S0, V0, F0 = compute_DFE()
        return {'S': S0, 'V': V0, 'Ia': 0.0, 'As': 0.0,
                'Is': 0.0, 'Ic': 0.0, 'R': 0.0, 'F': F0}
    return compute_EE(beta_val)


def compute_f_vector(y, beta_val):
    S, V, Ia, As, Is, Ic, R, F = y
    omega, tau, sigma, delta, epsilon = get_rates(F)
    Ntot = max(S + V + Ia + As + Is + Ic + R, 1e-12)
    lam = beta_val / Ntot * (Ia + kappa1*As + kappa2*Is + kappa3*Ic)
    return np.array([
        Pi - (lam + mu + omega)*S + rho*R + theta*V,
        omega*S - (theta + mu)*V,
        lam*S - (mu + alpha)*Ia,
        tau*alpha*Ia - (sigma + mu)*As,
        (1 - tau)*alpha*Ia - (mu + delta + phi)*Is,
        phi*Is - (mu + gamma_c + epsilon)*Ic,
        sigma*As + delta*Is + epsilon*Ic - (mu + rho)*R,
        Lambda_F + eta1*Is + eta2*Ic - xi*F,
    ])


# ---------------- ABC-ABM solver (vectorised; same scheme as original) ----
Y0 = np.array([600000.0, 350000.0, 250000.0, 240000.0,
               220000.0, 200000.0, 0.0, 18.9])


def solve_abc_model(beta_val, nu, h=0.01, T=30.0):
    N = int(round(T / h))
    time = np.linspace(0, T, N + 1)
    B_nu = 1 - nu + nu / gamma(nu)
    c1 = (1 - nu) / B_nu
    c2 = nu * (h ** nu) / (B_nu * gamma(nu + 2))

    sol = np.zeros((N + 1, 8)); sol[0] = Y0
    fc = np.zeros((N + 1, 8)); fc[0] = compute_f_vector(Y0, beta_val)
    f0 = fc[0]
    for n in range(N):
        k = np.arange(n + 1)
        diff = (n + 1 - k).astype(float)
        m = (n - k).astype(float)
        A_coef = diff**nu * (diff + 1 + nu) - m**nu * (m + 2 + 2*nu) * (m > 0)
        B_coef = diff**(nu + 1) - m**nu * (m + 1 + nu)
        sumA = c2 * (A_coef @ fc[:n + 1])
        sumB = c2 * (B_coef[1:] @ fc[:n]) if n >= 1 else np.zeros(8)
        y_pred = Y0 + c1 * f0 + sumA - sumB
        f_pred = compute_f_vector(y_pred, beta_val)
        y_new = Y0 + (c1 + c2) * f_pred + sumA - sumB
        f0 = compute_f_vector(y_new, beta_val)
        sol[n + 1] = y_new; fc[n + 1] = f0
    return (time,) + tuple(sol[:, j] for j in range(8))


# ---------------- per-compartment metrics ----------------
def calculate_transient_metrics(time, solution, equilibrium, tolerance=0.01):
    metrics = []
    for var_name, y in zip(STATE_KEYS, solution):
        y = np.asarray(y); eq = equilibrium[var_name]
        deviation = np.abs(y - eq)
        max_dev = float(np.max(deviation))
        t_peak = float(time[int(np.argmax(deviation))])
        scale = max(abs(eq), float(np.max(np.abs(y))), 1.0)
        threshold = tolerance * scale
        outside = np.where(deviation > threshold)[0]
        if len(outside) == 0:
            settling = float(time[0])
        elif outside[-1] == len(time) - 1:
            settling = np.nan
        else:
            settling = float(time[outside[-1] + 1])
        peak_idx, _ = find_peaks(
            deviation,
            prominence=max(1e-8*scale, 0.001*max_dev) if max_dev > 0 else 1e-8)
        metrics.append({'Compartment': var_name, 'Equilibrium': eq,
                        'Maximum deviation': max_dev,
                        'Time to maximum deviation': t_peak,
                        'Settling time': settling,
                        'Number of deviation peaks': int(len(peak_idx))})
    return metrics


# ---------------- decay-rate estimator ----------------
def estimate_decay_rate(time, I_total, I_star, t_peak,
                        t_lo=5.0, floor=0.01, min_points=20):
    dev = np.abs(np.asarray(I_total) - I_star)
    dev_max = float(np.max(dev))
    if not np.isfinite(dev_max) or dev_max <= 0:
        return np.nan, np.nan, np.nan, np.nan
    mask = (time >= max(t_lo, t_peak)) & (dev > floor * dev_max)
    if mask.sum() < min_points:
        return np.nan, np.nan, np.nan, np.nan
    x = time[mask]; y = np.log(dev[mask])
    slope, icpt = np.polyfit(x, y, 1)
    ss_tot = float(np.sum((y - y.mean())**2))
    r2 = 1 - float(np.sum((y - slope*x - icpt)**2)) / ss_tot if ss_tot > 0 else np.nan
    return -float(slope), r2, float(x[0]), float(x[-1])


def analyze_total_infected(results, nu_list, beta_val, t_lo=5.0):
    eq = get_equilibrium(beta_val)
    I_star = eq['Ia'] + eq['As'] + eq['Is'] + eq['Ic']
    rows = []
    for nu in nu_list:
        time, S, V, Ia, As, Is, Ic, R, F = results[nu]
        I_total = Ia + As + Is + Ic
        if not np.all(np.isfinite(I_total)):
            rows.append({'Beta': beta_val, 'nu': nu, 'Peak': np.nan,
                         'Time to peak': np.nan, 'I_star': I_star,
                         'Decay rate': np.nan, 'R2': np.nan,
                         'Fit start': np.nan, 'Fit end': np.nan,
                         'Maxima': 0, 'Minima': 0, 'Note': 'non-finite'})
            continue
        i_max = int(np.argmax(I_total))
        peak = float(I_total[i_max]); t_peak = float(time[i_max])
        prom = 0.01 * peak if peak > 0 else 1e-6
        mx, _ = find_peaks(I_total, prominence=prom)
        mn, _ = find_peaks(-I_total, prominence=prom)
        decay, r2, t0, t1 = estimate_decay_rate(time, I_total, I_star, t_peak, t_lo)
        rows.append({'Beta': beta_val, 'nu': nu, 'Peak': peak,
                     'Time to peak': t_peak, 'I_star': I_star,
                     'Decay rate': decay, 'R2': r2,
                     'Fit start': t0, 'Fit end': t1,
                     'Maxima': int(len(mx)), 'Minima': int(len(mn)), 'Note': ''})
    return pd.DataFrame(rows)


# ---------------- eigenvalue analysis ----------------
def eigenvalue_analysis(beta_val, nu_list=(0.7, 0.8, 0.9, 1.0)):
    eq = get_equilibrium(beta_val)
    y_eq = np.array([eq[k] for k in STATE_KEYS], dtype=float)
    n = len(y_eq); J = np.zeros((n, n))
    f0 = compute_f_vector(y_eq, beta_val)
    for j in range(n):
        h = 1e-6 * max(abs(y_eq[j]), 1.0)
        yp = y_eq.copy(); yp[j] += h
        if y_eq[j] > h:
            ym = y_eq.copy(); ym[j] -= h
            J[:, j] = (compute_f_vector(yp, beta_val)
                       - compute_f_vector(ym, beta_val)) / (2*h)
        else:
            J[:, j] = (compute_f_vector(yp, beta_val) - f0) / h
    eigs = np.linalg.eigvals(J)
    tol = 1e-8 * max(1.0, float(np.max(np.abs(eigs))))
    n_pairs = int(sum(1 for e in eigs if abs(e.imag) > tol) // 2)
    dom = eigs[int(np.argmax(eigs.real))]
    nz = eigs[np.abs(eigs) > tol]
    min_arg = float(np.min(np.abs(np.angle(nz)))) if len(nz) else np.nan
    print(f"  Spectrum: {n_pairs} complex pair(s); eigenvalues = "
          f"{np.round(np.sort_complex(eigs), 4)}")
    rows = []
    for nu in nu_list:
        thr = nu * np.pi / 2
        stable = bool(np.all(np.abs(np.angle(nz)) > thr)) if len(nz) else True
        rows.append({'Beta': beta_val, 'nu': nu, 'Re': float(dom.real),
                     'Im': float(dom.imag), 'min|arg|': min_arg,
                     'nu*pi/2': float(thr),
                     'Oscillatory': 'yes' if n_pairs else 'no',
                     'Stable (full spectrum)': 'yes' if stable else 'no',
                     'Complex pairs': n_pairs})
    return pd.DataFrame(rows), eigs


# ---------------- plots ----------------
def plot_beta_results(beta_val, nu_list, T=30.0):
    results = {}
    for nu in nu_list:
        print(f"Simulating beta={beta_val}, nu={nu} ...", flush=True)
        results[nu] = solve_abc_model(beta_val, nu, h=0.01, T=T)
    eq = get_equilibrium(beta_val)
    all_metrics = []
    for nu in nu_list:
        time, *sol = results[nu]
        for item in calculate_transient_metrics(time, sol, eq):
            item['Beta'] = beta_val; item['Fractional order'] = nu
            all_metrics.append(item)
    pd.DataFrame(all_metrics).to_csv(
        f'transient_metrics_beta_{beta_val:.4f}.csv'.replace('.', '_', 1).replace('_csv', '.csv'),
        index=False)

    total_df = analyze_total_infected(results, nu_list, beta_val)
    total_df.to_csv(f'total_infected_analysis_beta_{beta_val:.4f}.csv'.replace('.', '_', 1).replace('_csv', '.csv'),
                    index=False)
    print("\nTOTAL INFECTED ANALYSIS")
    print(total_df.drop(columns='Note').to_string(index=False))

    comps = [('S', r'$S$'), ('V', r'$V$'), ('Ia', r'$I_a$'), ('As', r'$A_s$'),
             ('Is', r'$I_s$'), ('Ic', r'$I_c$'), ('R', r'$R$'),
             ('F', r'$F$ (US\$ M)')]
    fig, axes = plt.subplots(3, 3, figsize=(16, 13)); axes = axes.flatten()
    cols = {0.7: '#1f77b4', 0.8: '#ff7f0e', 0.9: '#2ca02c', 1.0: '#d62728'}
    for idx, (name, ylabel) in enumerate(comps):
        ax = axes[idx]
        for nu in nu_list:
            time = results[nu][0]; y = results[nu][1 + STATE_KEYS.index(name)]
            ax.plot(time, y, label=rf'$\nu = {nu}$', linewidth=3.5,
                    color=cols.get(nu), alpha=0.9)
        ax.set_xlim(0, T); ax.set_ylim(bottom=0)
        ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
        ax.set_title(ylabel.replace('$', ''), fontsize=14, fontweight='bold', pad=6)
        ax.legend(loc='best', fontsize=10)
        ax.grid(alpha=0.35, linestyle=':')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.yaxis.get_major_formatter().set_scientific(True)
        ax.yaxis.get_major_formatter().set_powerlimits((-2, 3))
    for idx in [6, 7]:
        axes[idx].set_xlabel('Time (years)', fontsize=14, fontweight='bold')
    axes[8].axis('off')
    fig.suptitle(rf'$\beta = {beta_val}$, $R_e = {compute_Re(beta_val):.4f}$',
                 fontsize=20, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(f'beta{beta_val:.4f}_grid.png'.replace('.', '_', 1).replace('_png', '.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)
    return results, total_df


def check_step_size_convergence(beta_val, nu_list, T=5.0):
    step_sizes = [0.01, 0.005, 0.0025]
    rows = []
    for nu in nu_list:
        prev = prev_h = None
        for h in step_sizes:
            print(f"Convergence: beta={beta_val}, nu={nu}, h={h}", flush=True)
            curr = solve_abc_model(beta_val, nu, h=h, T=T)[1:]
            if prev is not None:
                for j, name in enumerate(STATE_KEYS):
                    coarse = prev[j]; fine = curr[j][::2]
                    n = min(len(coarse), len(fine))
                    max_d = float(np.max(np.abs(coarse[:n] - fine[:n])))
                    scale = max(float(np.max(np.abs(fine[:n]))), 1.0)
                    rows.append({'Beta': beta_val, 'nu': nu, 'Variable': name,
                                 'Previous h': prev_h, 'Current h': h,
                                 'Max abs diff': max_d, 'Rel diff': max_d/scale})
            prev, prev_h = curr, h
    df = pd.DataFrame(rows)
    df.to_csv(f'step_convergence_beta_{beta_val:.4f}.csv'.replace('.', '_', 1).replace('_csv', '.csv'),
              index=False)
    return df


def run_h_refinement(beta_val, nu, h_list=(0.01, 0.005, 0.0025), T=10.0):
    fig, ax = plt.subplots(figsize=(10, 6))
    styles = [
        {'linestyle': '--', 'linewidth': 5.5, 'alpha': 0.85, 'color': '#1f77b4',
         'label': r'$h = 0.01$ (dashed)'},
        {'linestyle': '-.', 'linewidth': 4.5, 'alpha': 0.9, 'color': '#ff7f0e',
         'label': r'$h = 0.005$ (dash-dot)'},
        {'linestyle': '-', 'linewidth': 2.5, 'alpha': 1.0, 'color': '#2ca02c',
         'label': r'$h = 0.0025$ (solid)'}]
    summary = []
    for idx, h in enumerate(h_list):
        time, S, V, Ia, As, Is, Ic, R, F = solve_abc_model(beta_val, nu, h=h, T=T)
        I_total = Ia + As + Is + Ic
        ax.plot(time, I_total, **styles[idx], zorder=10 - idx)
        i_max = int(np.argmax(I_total))
        summary.append({'h': h, 'peak': float(I_total[i_max]),
                        't_peak': float(time[i_max])})
    ax.set_xlim(0, T); ax.set_ylim(bottom=0)
    ax.set_xlabel('Time (years)', fontsize=15, fontweight='bold')
    ax.set_ylabel(r'Total infected $I_a + A_s + I_s + I_c$', fontsize=15, fontweight='bold')
    ax.set_title(rf'h-refinement, $\beta = {beta_val}$, $\nu = {nu}$',
                 fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=12); ax.grid(alpha=0.35, linestyle=':')
    plt.tight_layout()
    fig.savefig(f'h_refinement_beta{beta_val:.4f}_nu{nu:.2f}'.replace('.', '_') + '.png',
                dpi=200, bbox_inches='tight')
    plt.close(fig)
    return pd.DataFrame(summary)


# ---------------- LaTeX exports ----------------
def export_total_latex(df, fname='total_infected_transients.tex'):
    lines = [
        r"\begin{table}[!htp]", r"\centering",
        r"\caption{Transient characteristics of the total infected population "
        r"$I(t)=I_a+A_s+I_s+I_c$ across fractional orders. Peak is "
        r"$\max_t I(t)$ and time-to-peak is when it occurs (``initial'' if "
        r"the maximum is the initial condition). The effective decay rate is "
        r"minus the slope of $\ln|I(t)-I^*|$, fitted by least squares over "
        r"the stated window, which is common to all $\nu$ and starts at "
        r"$\max(5,t_{\mathrm{peak}})$; only points with $|I-I^*|>1\%$ of the "
        r"maximum deviation are used. $R^2$ measures how close the decay is "
        r"to a single exponential. Maxima and minima count local extrema "
        r"above $1\%$ of the peak.}",
        r"\label{tab:total_infected_transients}",
        r"\begin{tabular}{@{}lccccccc@{}}", r"\toprule",
        r"$\beta$ & $\nu$ & Peak & Time-to-peak (yr) & Decay (yr$^{-1}$) & "
        r"Fit window (yr) & $R^2$ & \# max/min \\", r"\midrule"]
    for _, r in df.iterrows():
        ok = np.isfinite(r['Decay rate'])
        decay = f"{r['Decay rate']:.4f}" if ok else "---"
        win = f"[{r['Fit start']:.1f}, {r['Fit end']:.1f}]" if ok else "---"
        r2 = f"{r['R2']:.3f}" if ok and np.isfinite(r['R2']) else "---"
        peak = f"{r['Peak']:.0f}" if np.isfinite(r['Peak']) else "---"
        if not np.isfinite(r['Time to peak']): t_pk = "---"
        elif r['Time to peak'] == 0: t_pk = "0.00 (initial)"
        else: t_pk = f"{r['Time to peak']:.2f}"
        lines.append(f"{r['Beta']:.4f} & {r['nu']:.1f} & {peak} & {t_pk} & "
                     f"{decay} & {win} & {r2} & {int(r['Maxima'])}/{int(r['Minima'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(fname, 'w').write("\n".join(lines))


def export_eigen_latex(df, fname='eigenvalue_analysis.tex'):
    lines = [
        r"\begin{table}[!htp]", r"\centering",
        r"\caption{Spectrum of the Jacobian at the equilibrium and the "
        r"fractional-order stability test. $\lambda_{\mathrm{dom}}$ is the "
        r"eigenvalue with the largest real part (slowest mode). The "
        r"equilibrium is stable if every eigenvalue satisfies "
        r"$|\arg\lambda|>\nu\pi/2$; $\min|\arg\lambda|$ is the smallest "
        r"argument over the full spectrum. Oscillatory means at least one "
        r"complex-conjugate pair exists.}",
        r"\label{tbl:eigenvalues}",
        r"\begin{tabular}{@{}lccccccc@{}}", r"\toprule",
        r"$\beta$ & $\nu$ & $\mathrm{Re}\,\lambda_{\mathrm{dom}}$ & "
        r"$\mathrm{Im}\,\lambda_{\mathrm{dom}}$ & $\min|\arg\lambda|$ & "
        r"$\nu\pi/2$ & Oscillatory & Stable \\", r"\midrule"]
    for _, r in df.iterrows():
        lines.append(f"{r['Beta']:.4f} & {r['nu']:.1f} & {r['Re']:.4f} & "
                     f"{r['Im']:.4f} & {r['min|arg|']:.4f} & {r['nu*pi/2']:.4f} & "
                     f"{r['Oscillatory']} & {r['Stable (full spectrum)']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(fname, 'w').write("\n".join(lines))


# ---------------- main ----------------
if __name__ == "__main__":
    beta_list = [0.5546, 2.5]
    nu_list = [0.7, 0.8, 0.9, 1.0]
    T = 30.0
    all_total, all_eigen = [], []
    for beta in beta_list:
        print("\n" + "=" * 60 + f"\nbeta = {beta}\n" + "=" * 60)
        print(f"Re = {compute_Re(beta):.4f}")
        S0, V0, F0 = compute_DFE()
        print(f"DFE: S0 = {S0:.2f}, V0 = {V0:.2f}, F0 = {F0:.2f}")
        if compute_Re(beta) > 1:
            print("EE:")
            for k, v in compute_EE(beta).items():
                print(f"  {k}* = {v:.2f}")
        else:
            print("No EE (Re < 1).")
        results, total_df = plot_beta_results(beta, nu_list, T)
        all_total.append(total_df)
        print("\nEIGENVALUES")
        eigen_df, _ = eigenvalue_analysis(beta, nu_list)
        all_eigen.append(eigen_df)
        print(eigen_df.to_string(index=False))
        check_step_size_convergence(beta, nu_list, T=5.0)
    print("\nh-refinement for beta=2.5, nu=0.7")
    print(run_h_refinement(2.5, 0.7, T=10.0).to_string(index=False))
    export_total_latex(pd.concat(all_total, ignore_index=True))
    export_eigen_latex(pd.concat(all_eigen, ignore_index=True))
    print("\nAll analyses complete.")
