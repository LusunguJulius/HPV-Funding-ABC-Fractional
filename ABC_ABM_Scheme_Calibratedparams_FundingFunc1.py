"""
Quantitative transient analysis across fractional orders (Comment 23).

Visual upgrade:
  - Global rcParams: bold labels, bold titles, thicker axes, thicker ticks.
  - All curves plotted with increased linewidth (>= 3.5) for visibility.
  - h_refinement keeps distinct line styles, but all bold and thick.
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.special import gamma
from scipy.optimize import root
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# ------------------------------------------------------------
# GLOBAL PLOT STYLING (applied to every figure)
# ------------------------------------------------------------
plt.rcParams.update({
    'font.size':           13,
    'axes.labelweight':    'bold',
    'axes.titleweight':    'bold',
    'axes.linewidth':      1.5,
    'xtick.major.width':   1.5,
    'ytick.major.width':   1.5,
    'xtick.major.size':    6,
    'ytick.major.size':    6,
    'legend.frameon':      True,
    'legend.framealpha':   0.9,
    'legend.edgecolor':    'black',
    'legend.fontsize':     11,
    'lines.linewidth':     3.5,      # default for every plt.plot
})


# ------------------------------------------------------------
# Baseline parameters
# ------------------------------------------------------------
Pi         = 186000.0
mu         = 0.1
gamma_c    = 0.005
rho        = 0.25
alpha      = 0.5013
phi        = 0.5926
theta      = 0.1
kappa1     = 0.16
kappa2     = 0.8383
kappa3     = 0.6427
omega_base = 0.4820
sigma_base = 0.9405
delta_base = 0.0500
epsilon_base = 0.0133
tau_base   = 0.9454
beta1      = 0.02
beta2      = 0.001
Lambda_F   = 18.9
eta1       = 1.98e-7
eta2       = 9.23e-6
xi         = 1.0


# ------------------------------------------------------------
# Funding-dependent rates
# ------------------------------------------------------------
def psi(k, F):
    return k * (1 + beta1 * F / (1 + beta2 * F))


def get_rates(F):
    omega   = psi(omega_base, F)
    tau     = min(psi(tau_base, F), 1.0)
    sigma   = psi(sigma_base, F)
    delta   = psi(delta_base, F)
    epsilon = psi(epsilon_base, F)
    return omega, tau, sigma, delta, epsilon


# ------------------------------------------------------------
# Reproduction number and equilibria
# ------------------------------------------------------------
def compute_Re(beta_val):
    F0 = Lambda_F / xi
    omega0, tau0, sigma0, delta0, epsilon0 = get_rates(F0)
    d1 = mu + alpha
    d2 = mu + sigma0
    d3 = mu + delta0 + phi
    d4 = mu + gamma_c + epsilon0
    factor = beta_val * (theta + mu) / (theta + mu + omega0)
    return factor * (
        1.0 / d1
        + kappa1 * tau0 * alpha / (d1 * d2)
        + kappa2 * (1 - tau0) * alpha / (d1 * d3)
        + kappa3 * phi * (1 - tau0) * alpha / (d1 * d3 * d4)
    )


def compute_DFE():
    F0 = Lambda_F / xi
    omega0, _, _, _, _ = get_rates(F0)
    S0 = Pi * (mu + theta) / (mu * (mu + theta + omega0))
    V0 = Pi * omega0 / (mu * (mu + theta + omega0))
    return S0, V0, F0


def compute_EE(beta_val):
    Re = compute_Re(beta_val)
    if Re <= 1:
        return None
    F0 = Lambda_F / xi

    def eq_F(F):
        omega, tau, sigma, delta, epsilon = get_rates(F)
        d2 = mu + sigma
        d3 = mu + delta + phi
        d4 = mu + gamma_c + epsilon
        m1 = tau * alpha / d2
        m2 = (1 - tau) * alpha / d3
        m3 = phi * (1 - tau) * alpha / (d3 * d4)
        K = 1 + m1 + m2 + m3 + (sigma*m1 + delta*m2 + epsilon*m3) / (mu + rho)
        Ia = (Re - 1) / (K * Re) * (Pi / mu)
        Is = m2 * Ia
        Ic = m3 * Ia
        return F - (Lambda_F + eta1*Is + eta2*Ic) / xi

    sol = root(eq_F, F0)
    if not sol.success:
        raise ValueError("Could not solve for F*")
    F_star = float(sol.x[0])

    omega, tau, sigma, delta, epsilon = get_rates(F_star)
    d2 = mu + sigma
    d3 = mu + delta + phi
    d4 = mu + gamma_c + epsilon
    m1 = tau * alpha / d2
    m2 = (1 - tau) * alpha / d3
    m3 = phi * (1 - tau) * alpha / (d3 * d4)
    A = (theta + mu + omega) / (theta + mu)
    K = 1 + m1 + m2 + m3 + (sigma*m1 + delta*m2 + epsilon*m3) / (mu + rho)
    Ia_star = (Re - 1) / (K * Re) * (Pi / mu)
    S_star  = (Pi / mu) / (A * Re)
    V_star  = omega / (theta + mu) * S_star
    return {
        'S': S_star, 'V': V_star,
        'Ia': Ia_star,
        'As': m1 * Ia_star,
        'Is': m2 * Ia_star,
        'Ic': m3 * Ia_star,
        'R': (sigma*m1 + delta*m2 + epsilon*m3) / (mu + rho) * Ia_star,
        'F': F_star,
    }


def get_equilibrium(beta_val):
    if compute_Re(beta_val) <= 1:
        S_star, V_star, F_star = compute_DFE()
        return {'S': S_star, 'V': V_star, 'Ia': 0.0, 'As': 0.0,
                'Is': 0.0, 'Ic': 0.0, 'R': 0.0, 'F': F_star}
    return compute_EE(beta_val)


# ------------------------------------------------------------
# RHS
# ------------------------------------------------------------
def compute_f_vector(y, beta_val):
    S, V, Ia, As, Is, Ic, R, F = y
    omega, tau, sigma, delta, epsilon = get_rates(F)
    Ntot = S + V + Ia + As + Is + Ic + R
    if Ntot <= 1e-12:
        Ntot = 1e-12
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


# ------------------------------------------------------------
# ABC-ABM solver (anchored at y0)
# ------------------------------------------------------------
def solve_abc_model(beta_val, nu, h=0.01, T=30.0):
    Nsteps = int(round(T / h))
    time = np.linspace(0, T, Nsteps + 1)

    y0 = np.array([600000.0, 350000.0, 250000.0, 240000.0,
                   220000.0, 200000.0, 0.0, 18.9], dtype=float)

    B_nu = 1 - nu + nu / gamma(nu)
    c1   = (1 - nu) / B_nu
    c2   = nu * (h ** nu) / (B_nu * gamma(nu + 2))

    S  = np.zeros(Nsteps + 1); V  = np.zeros(Nsteps + 1)
    Ia = np.zeros(Nsteps + 1); As = np.zeros(Nsteps + 1)
    Is = np.zeros(Nsteps + 1); Ic = np.zeros(Nsteps + 1)
    R  = np.zeros(Nsteps + 1); F  = np.zeros(Nsteps + 1)

    S[0], V[0], Ia[0], As[0], Is[0], Ic[0], R[0], F[0] = y0

    f_cache = np.zeros((Nsteps + 1, 8))
    f0      = compute_f_vector(y0, beta_val)
    f_cache[0] = f0

    for n in range(Nsteps):
        sumA = np.zeros(8)
        sumB = np.zeros(8)
        for k in range(n + 1):
            diff = n + 1 - k
            A_coef = diff ** nu * (diff + 1 + nu)
            if n - k > 0:
                A_coef -= (n - k) ** nu * (n - k + 2 + 2 * nu)
            sumA += A_coef * f_cache[k]
            if k >= 1:
                if n - k == 0:
                    B_coef = diff ** (nu + 1)
                else:
                    B_coef = diff ** (nu + 1) - (n - k) ** nu * (n - k + 1 + nu)
                sumB += B_coef * f_cache[k - 1]
        sumA *= c2
        sumB *= c2

        y_pred = y0 + c1 * f0 + sumA - sumB
        f_pred = compute_f_vector(y_pred, beta_val)

        y_new = y0 + (c1 + c2) * f_pred + sumA - sumB
        f_new = compute_f_vector(y_new, beta_val)

        S[n+1], V[n+1], Ia[n+1], As[n+1], Is[n+1], Ic[n+1], R[n+1], F[n+1] = y_new
        f_cache[n+1] = f_new
        f0 = f_new

    return time, S, V, Ia, As, Is, Ic, R, F


# ------------------------------------------------------------
# Per-compartment transient metrics
# ------------------------------------------------------------
def calculate_transient_metrics(time, solution, equilibrium, tolerance=0.01):
    names = ['S', 'V', 'Ia', 'As', 'Is', 'Ic', 'R', 'F']
    metrics = []
    for var_name, y in zip(names, solution):
        y  = np.asarray(y)
        eq = equilibrium[var_name]
        deviation = np.abs(y - eq)
        max_dev = float(np.max(deviation))
        t_peak  = float(time[int(np.argmax(deviation))])

        scale     = max(abs(eq), float(np.max(np.abs(y))), 1.0)
        threshold = tolerance * scale

        outside = np.where(deviation > threshold)[0]
        if len(outside) == 0:
            settling = float(time[0])
        elif outside[-1] == len(time) - 1:
            settling = np.nan
        else:
            settling = float(time[outside[-1] + 1])

        peak_indices, _ = find_peaks(
            deviation,
            prominence=max(1e-8 * scale, 0.001 * max_dev) if max_dev > 0 else 1e-8
        )

        if len(peak_indices) >= 2:
            amps = deviation[peak_indices]
            pts  = time[peak_indices]
            valid = amps[1:] > 0
            if np.any(valid):
                rates = np.log(amps[:-1][valid] / amps[1:][valid]) / np.diff(pts)[valid]
                decay = float(np.mean(rates))
            else:
                decay = np.nan
        else:
            decay = np.nan

        metrics.append({
            'Compartment': var_name,
            'Equilibrium': eq,
            'Maximum deviation': max_dev,
            'Time to maximum deviation': t_peak,
            'Settling time': settling,
            'Number of deviation peaks': int(len(peak_indices)),
            'Estimated decay rate': decay,
        })
    return metrics


# ------------------------------------------------------------
# Total infected analysis
# ------------------------------------------------------------
def analyze_total_infected(results, nu_list, beta_val):
    rows = []
    for nu in nu_list:
        time, S, V, Ia, As, Is, Ic, R, F = results[nu]
        I_total = Ia + As + Is + Ic

        if not np.all(np.isfinite(I_total)):
            rows.append({
                'Beta': beta_val, 'nu': nu, 'Peak': np.nan,
                'Time to peak': np.nan, 'Decay rate': np.nan,
                'Maxima': 0, 'Minima': 0,
                'Note': 'non-finite trajectory'
            })
            continue

        i_max  = int(np.argmax(I_total))
        peak   = float(I_total[i_max])
        t_peak = float(time[i_max])

        prom = 0.01 * peak if peak > 0 else 1e-6
        mx, _ = find_peaks( I_total, prominence=prom)
        mn, _ = find_peaks(-I_total, prominence=prom)

        if t_peak > 0:
            mask = time > 1.5 * t_peak
            if mask.sum() > 20:
                slope, _ = np.polyfit(time[mask],
                                      np.log(np.maximum(I_total[mask], 1e-12)), 1)
                decay = -float(slope)
            else:
                decay = np.nan
        else:
            decay = np.nan

        rows.append({
            'Beta': beta_val, 'nu': nu,
            'Peak': peak, 'Time to peak': t_peak,
            'Decay rate': decay,
            'Maxima': int(len(mx)), 'Minima': int(len(mn)),
            'Note': '',
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------
# Eigenvalue analysis
# ------------------------------------------------------------
def eigenvalue_analysis(beta_val, nu_list=(0.7, 0.8, 0.9, 1.0)):
    eq   = get_equilibrium(beta_val)
    keys = ['S', 'V', 'Ia', 'As', 'Is', 'Ic', 'R', 'F']
    y_eq = np.array([eq[k] for k in keys], dtype=float)

    n   = len(y_eq)
    J   = np.zeros((n, n))
    eps = 1e-6
    f0  = compute_f_vector(y_eq, beta_val)
    for j in range(n):
        yp = y_eq.copy(); yp[j] += eps
        J[:, j] = (compute_f_vector(yp, beta_val) - f0) / eps

    eigs = np.linalg.eigvals(J)
    args = np.abs(np.angle(eigs))

    n_complex_pairs = sum(1 for e in eigs if abs(e.imag) > 1e-8) // 2
    print(f"  Spectrum: {n_complex_pairs} complex-conjugate pair(s); "
          f"eigenvalues = {np.round(eigs, 4)}")

    rows = []
    for nu in nu_list:
        thr    = nu * np.pi / 2
        margin = args - thr
        dom    = eigs[int(np.argmin(margin))]
        rows.append({
            'Beta': beta_val, 'nu': nu,
            'Re': float(dom.real), 'Im': float(dom.imag),
            '|arg|': float(abs(np.angle(dom))),
            'nu*pi/2': float(thr),
            'Oscillatory (dominant)': 'yes' if abs(dom.imag) > 1e-8 else 'no',
            'Stable (dominant)': 'yes' if dom.real < 0 else 'no',
            'Complex pairs (full spectrum)': n_complex_pairs,
        })
    return pd.DataFrame(rows), eigs


# ------------------------------------------------------------
# Plot grid (bold, thick, all axes from 0)
# ------------------------------------------------------------
def plot_beta_results(beta_val, nu_list, T=30.0):
    results = {}
    for nu in nu_list:
        print(f"Simulating beta={beta_val}, nu={nu} ...")
        results[nu] = solve_abc_model(beta_val, nu, h=0.01, T=T)
    print("Simulations complete.")

    eq = get_equilibrium(beta_val)
    all_metrics = []
    for nu in nu_list:
        time, S, V, Ia, As, Is, Ic, R, F = results[nu]
        metrics = calculate_transient_metrics(
            time, [S, V, Ia, As, Is, Ic, R, F], eq, tolerance=0.01)
        for item in metrics:
            item['Beta'] = beta_val
            item['Fractional order'] = nu
            all_metrics.append(item)

    metrics_df   = pd.DataFrame(all_metrics)
    metrics_file = f'transient_metrics_beta_{beta_val:.4f}.csv'.replace('.', '_')
    metrics_df.to_csv(metrics_file, index=False)
    print(f"Per-compartment metrics saved to {metrics_file}")

    total_df   = analyze_total_infected(results, nu_list, beta_val)
    total_file = f'total_infected_analysis_beta_{beta_val:.4f}.csv'.replace('.', '_')
    total_df.to_csv(total_file, index=False)
    print("\nTOTAL INFECTED ANALYSIS")
    print(total_df.to_string(index=False))
    print(f"Saved to {total_file}")

    # 3x3 grid — bold, thick, origin anchored
    comps = [('S', r'$S$'), ('V', r'$V$'), ('Ia', r'$I_a$'),
             ('As', r'$A_s$'), ('Is', r'$I_s$'), ('Ic', r'$I_c$'),
             ('R', r'$R$'), ('F', r'$F$ (US\$ M)')]
    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    axes = axes.flatten()

    # Distinct, high-contrast colors per nu
    nu_colors = {0.7: '#1f77b4', 0.8: '#ff7f0e', 0.9: '#2ca02c', 1.0: '#d62728'}

    for idx, (var_name, ylabel) in enumerate(comps):
        ax = axes[idx]
        for nu in nu_list:
            time, S, V, Ia, As, Is, Ic, R, F = results[nu]
            y = {'S':S,'V':V,'Ia':Ia,'As':As,'Is':Is,'Ic':Ic,'R':R,'F':F}[var_name]
            ax.plot(time, y,
                    label=rf'$\nu = {nu}$',
                    linewidth=3.5,
                    color=nu_colors.get(nu, None),
                    alpha=0.9)

        ax.set_xlim(0, T)
        ax.set_ylim(bottom=0)
        ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
        ax.set_title(ylabel.replace('$', ''), fontsize=14, fontweight='bold',
                     pad=6)
        ax.legend(loc='best', fontsize=10, frameon=True,
                  framealpha=0.9, edgecolor='black')
        ax.grid(alpha=0.35, linestyle=':')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.yaxis.get_major_formatter().set_scientific(True)
        ax.yaxis.get_major_formatter().set_powerlimits((-2, 3))
        ax.tick_params(axis='both', labelsize=11, width=1.5)

    for idx in [6, 7]:
        axes[idx].set_xlabel('Time (years)', fontsize=14, fontweight='bold')
    axes[8].axis('off')

    Re = compute_Re(beta_val)
    fig.suptitle(rf'$\beta = {beta_val}$, $R_e = {Re:.4f}$',
                 fontsize=20, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig_file = f'beta{beta_val:.4f}_grid.png'.replace('.', '_')
    fig.savefig(fig_file, dpi=300, bbox_inches='tight')
    print(f"Figure saved as '{fig_file}'")
    plt.close(fig)

    return results, total_df


# ------------------------------------------------------------
# Step-size convergence (numeric table)
# ------------------------------------------------------------
def check_step_size_convergence(beta_val, nu_list, T=5.0):
    step_sizes = [0.01, 0.005, 0.0025]
    names = ['S','V','Ia','As','Is','Ic','R','F']
    rows = []
    for nu in nu_list:
        prev = None
        prev_h = None
        for h in step_sizes:
            print(f"Convergence: beta={beta_val}, nu={nu}, h={h}")
            result = solve_abc_model(beta_val, nu, h=h, T=T)
            curr = result[1:]
            if prev is not None:
                for j, name in enumerate(names):
                    coarse = prev[j]
                    fine   = curr[j][::2]
                    n = min(len(coarse), len(fine))
                    diff = np.abs(coarse[:n] - fine[:n])
                    max_d = float(np.max(diff))
                    scale = max(float(np.max(np.abs(fine[:n]))), 1.0)
                    rows.append({
                        'Beta': beta_val, 'nu': nu, 'Variable': name,
                        'Previous h': prev_h, 'Current h': h,
                        'Max abs diff': max_d,
                        'Rel diff': max_d / scale,
                    })
            prev, prev_h = curr, h
    df = pd.DataFrame(rows)
    fname = f'step_convergence_beta_{beta_val:.4f}.csv'.replace('.', '_')
    df.to_csv(fname, index=False)
    print(f"Step-size convergence saved to {fname}")
    return df


# ------------------------------------------------------------
# h-refinement figure (bold, thick, distinct styles)
# ------------------------------------------------------------
def run_h_refinement(beta_val, nu, h_list=(0.01, 0.005, 0.0025), T=10.0):
    fig, ax = plt.subplots(figsize=(10, 6))

    styles = [
        {'linestyle': '--', 'linewidth': 5.5, 'alpha': 0.85,
         'color': '#1f77b4', 'label': r'$h = 0.01$ (dashed)'},
        {'linestyle': '-.', 'linewidth': 4.5, 'alpha': 0.9,
         'color': '#ff7f0e', 'label': r'$h = 0.005$ (dash-dot)'},
        {'linestyle': '-',  'linewidth': 2.5, 'alpha': 1.0,
         'color': '#2ca02c', 'label': r'$h = 0.0025$ (solid)'},
    ]

    summary = []
    for idx, h in enumerate(h_list):
        time, S, V, Ia, As, Is, Ic, R, F = solve_abc_model(beta_val, nu, h=h, T=T)
        I_total = Ia + As + Is + Ic
        ax.plot(time, I_total, **styles[idx], zorder=10 - idx)
        i_max = int(np.argmax(I_total))
        summary.append({'h': h,
                        'peak': float(I_total[i_max]),
                        't_peak': float(time[i_max])})

    ax.set_xlim(0, T)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Time (years)', fontsize=15, fontweight='bold')
    ax.set_ylabel(r'Total infected $I_a + A_s + I_s + I_c$',
                  fontsize=15, fontweight='bold')
    ax.set_title(rf'h-refinement, $\beta = {beta_val}$, $\nu = {nu}$',
                 fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=12, frameon=True,
              framealpha=0.9, edgecolor='black')
    ax.grid(alpha=0.35, linestyle=':')
    ax.tick_params(axis='both', labelsize=12, width=1.5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout()
    fname = f'h_refinement_beta{beta_val:.4f}_nu{nu:.2f}'.replace('.', '_')
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"h-refinement figure saved to {fname}")
    return pd.DataFrame(summary)


# ------------------------------------------------------------
# LaTeX exports
# ------------------------------------------------------------
def export_total_latex(df, fname='total_infected_transients.tex'):
    lines = [
        r"\begin{table}[!htp]", r"\centering",
        r"\caption{Transient characteristics of the total infected population "
        r"$I(t) = I_a + A_s + I_s + I_c$ across fractional orders and "
        r"transmission scenarios. Peak is $\max_t I(t)$; time-to-peak is the "
        r"time at which it occurs (``initial'' when the maximum is the initial "
        r"condition); decay rate is the slope of $\ln I(t)$ fitted from "
        r"$1.5\,t_{\mathrm{peak}}$ onwards (positive = decay); maxima and "
        r"minima count local extrema above $1\%$ of the peak.}",
        r"\label{tab:total_infected_transients}",
        r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
        r"$\beta$ & $\nu$ & Peak & Time-to-peak (yr) & Decay (yr$^{-1}$) "
        r"& \# maxima & \# minima \\", r"\midrule",
    ]
    for _, r in df.iterrows():
        decay = f"{r['Decay rate']:.4f}" if np.isfinite(r['Decay rate']) else "---"
        peak  = f"{r['Peak']:.0f}"     if np.isfinite(r['Peak'])     else "---"
        if not np.isfinite(r['Time to peak']):
            t_pk = "---"
        elif r['Time to peak'] == 0:
            t_pk = "0.00 (initial)"
        else:
            t_pk = f"{r['Time to peak']:.2f}"
        lines.append(f"{r['Beta']:.4f} & {r['nu']:.1f} & {peak} & {t_pk} & "
                     f"{decay} & {r['Maxima']} & {r['Minima']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(fname, 'w') as f:
        f.write("\n".join(lines))
    print(f"LaTeX table saved to {fname}")


def export_eigen_latex(df, fname='eigenvalue_analysis.tex'):
    lines = [
        r"\begin{table}[!htp]", r"\centering",
        r"\caption{Dominant eigenvalue of the Jacobian at the equilibrium and "
        r"the fractional-order stability threshold $\nu\pi/2$ for each "
        r"fractional order. A stable equilibrium requires "
        r"$|\arg\lambda|>\nu\pi/2$. A complex dominant eigenvalue indicates a "
        r"damped oscillation (model-generated); a real one indicates a "
        r"monotone return.}",
        r"\label{tbl:eigenvalues}",
        r"\begin{tabular}{@{}lccccccc@{}}", r"\toprule",
        r"$\beta$ & $\nu$ & $\mathrm{Re}\,\lambda$ & $\mathrm{Im}\,\lambda$ & "
        r"$|\arg\lambda|$ & $\nu\pi/2$ & Oscillatory & Stable \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        lines.append(f"{r['Beta']:.4f} & {r['nu']:.1f} & {r['Re']:.4f} & "
                     f"{r['Im']:.4f} & {r['|arg|']:.4f} & "
                     f"{r['nu*pi/2']:.4f} & {r['Oscillatory (dominant)']} & "
                     f"{r['Stable (dominant)']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(fname, 'w') as f:
        f.write("\n".join(lines))
    print(f"LaTeX table saved to {fname}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    beta_list = [0.5546, 2.5]
    nu_list   = [0.7, 0.8, 0.9, 1.0]
    T = 30.0

    all_total = []
    all_eigen = []

    for beta in beta_list:
        print("\n" + "=" * 60)
        print(f"beta = {beta}")
        print("=" * 60)
        print(f"Re = {compute_Re(beta):.4f}")

        S0, V0, F0 = compute_DFE()
        print(f"DFE: S0 = {S0:.2f}, V0 = {V0:.2f}, F0 = {F0:.2f}")

        if compute_Re(beta) > 1:
            EE = compute_EE(beta)
            print("EE:")
            for k, v in EE.items():
                print(f"  {k}* = {v:.2f}")
        else:
            print("No EE (Re < 1).")

        results, total_df = plot_beta_results(beta, nu_list, T)
        all_total.append(total_df)

        print("\nEIGENVALUES")
        eigen_df, eigs = eigenvalue_analysis(beta, nu_list)
        all_eigen.append(eigen_df)
        print(eigen_df.to_string(index=False))

        check_step_size_convergence(beta_val=beta, nu_list=nu_list, T=5.0)

    print("\nh-refinement for beta=2.5, nu=0.7")
    print(run_h_refinement(2.5, 0.7, T=10.0).to_string(index=False))

    export_total_latex(pd.concat(all_total, ignore_index=True))
    export_eigen_latex(pd.concat(all_eigen, ignore_index=True))

    print("\nAll analyses complete.")