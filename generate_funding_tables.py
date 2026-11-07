"""
generate_funding_tables.py

Self-contained script that:
  (1) Defines the full parameter set (matching the corrected simulation).
  (2) Implements the funding response with the cap:
          m(F)       = 1 + beta1*F / (1 + beta2*F)
          Psi_tau(F) = tau0 * m(F)
          tau(F)     = min{1, Psi_tau(F)}   <-- the cap
  (3) Computes R_e(F) with the cap applied.
  (4) Writes two LaTeX tables:
          tau_cap_table.tex        - screening proportion and the cap
          funding_impl_table.tex   - all funding-dependent params + R_e
  (5) Prints the tables and a diagnostics summary to stdout.

Run with:
    python generate_funding_tables.py
"""

import os

# ==============================================================================
# 1. Parameters (identical to the corrected simulation code)
# ==============================================================================
params = {
    # epidemiological
    'beta':   2.5,         # high-contact regime (endemic scenario, Section 9.3)
    'mu':     0.1,
    'gamma':  0.005,
    'alpha':  0.5013,
    'phi':    0.5926,
    'theta':  0.1,
    'kappa1': 0.16,
    'kappa2': 0.8383,
    'kappa3': 0.6427,
    # funding-dependent baseline values
    'omega0': 0.4820,
    'sigma0': 0.9405,
    'delta0': 0.0500,
    'eps0':   0.0133,
    'tau0':   0.9454,
    # funding response parameters (illustrative values for Section 9.3)
    'beta1':  0.02,
    'beta2':  0.001,
}

# Funding levels (million USD) reported in the tables
F_levels = [0.0, 10.0, 18.9, 20.0, 40.0]

# ==============================================================================
# 2. Funding response with cap
# ==============================================================================
def m(F, p):
    """Funding multiplier m(F) = 1 + beta1*F / (1 + beta2*F)."""
    return 1.0 + p['beta1'] * F / (1.0 + p['beta2'] * F)

def psi_tau(F, p):
    """Uncapped screening response Psi_tau(F) = tau0 * m(F)."""
    return p['tau0'] * m(F, p)

def tau(F, p):
    """Capped screening proportion tau(F) = min{1, Psi_tau(F)}."""
    return min(1.0, psi_tau(F, p))

def F_crit(p):
    """Funding level at which the cap binds (None if it never binds).

    Solves tau0*(1 + beta1*F/(1 + beta2*F)) = 1:
        F_crit = (1 - tau0) / (tau0*beta1 - (1 - tau0)*beta2),
    provided the denominator is positive.
    """
    denom = p['tau0'] * p['beta1'] - (1.0 - p['tau0']) * p['beta2']
    if denom <= 0:
        return None
    return (1.0 - p['tau0']) / denom

def compute_Re(F, p):
    """Effective reproduction number with capped tau(F).

    R_e = [beta*(theta+mu)/(theta+mu+omega(F))] *
          [1/d1 + kappa1*tau*alpha/(d1*d2)
                + kappa2*(1-tau)*alpha/(d1*d3)
                + kappa3*phi*(1-tau)*alpha/(d1*d3*d4)]
    with d1=mu+alpha, d2=mu+sigma(F), d3=mu+delta(F)+phi, d4=mu+gamma+eps(F).
    """
    factor  = m(F, p)
    omega   = p['omega0'] * factor
    tau_c   = tau(F, p)                       # <-- cap applied here
    sigma   = p['sigma0'] * factor
    delta   = p['delta0'] * factor
    eps     = p['eps0']   * factor

    d1 = p['mu'] + p['alpha']
    d2 = p['mu'] + sigma
    d3 = p['mu'] + delta + p['phi']
    d4 = p['mu'] + p['gamma'] + eps

    prefac = p['beta'] * (p['theta'] + p['mu']) / (p['theta'] + p['mu'] + omega)
    term1  = 1.0 / d1
    term2  = p['kappa1'] * tau_c * p['alpha'] / (d1 * d2)
    term3  = p['kappa2'] * (1.0 - tau_c) * p['alpha'] / (d1 * d3)
    term4  = p['kappa3'] * p['phi'] * (1.0 - tau_c) * p['alpha'] / (d1 * d3 * d4)
    return max(0.0, prefac * (term1 + term2 + term3 + term4))

# ==============================================================================
# 3. Build Table 1: tau-cap table
# ==============================================================================
def build_tau_cap_table(p, F_vals):
    rows = []
    for F in F_vals:
        rows.append(
            f"${F:g}$ & ${m(F, p):.3f}$ & ${psi_tau(F, p):.4f}$ & "
            f"${tau(F, p):.4f}$ & ${1.0 - tau(F, p):.4f}$ \\\\"
        )
    body = "\n".join(rows)

    Fc = F_crit(p)
    if Fc is None:
        cap_note = (
            "With the calibrated parameters the denominator of the threshold "
            "expression is non-positive, so the cap never binds on "
            "$[0,F_{\\max}]$."
        )
    else:
        cap_note = (
            f"The uncapped response $\\Psi_\\tau(F)$ exceeds $1$ for "
            f"$F\\ge F_{{\\mathrm{{crit}}}}\\approx {Fc:.1f}$ million USD, "
            f"at which point the cap binds and the flow "
            f"$(1-\\tau(F))\\alpha I_a$ from $I_a$ to $I_s$ vanishes identically."
        )

    return (
        r"""\begin{table}[!htp]
\centering
\caption{Effect of the cap on the screening proportion
$\tau(F)=\min\{1,\Psi_\tau(F)\}$ at selected funding levels, using the
illustrative parameters $\beta_1=%(beta1)g$, $\beta_2=%(beta2)g$, and the
calibrated baseline $\tau=%(tau0)g$. %(cap_note)s}
\label{tbl:tau_cap}
\begin{tabular}{@{}ccccc@{}}
\toprule
$F$ (million USD) & $m(F)$ & $\Psi_\tau(F)$ (uncapped) & $\tau(F)=\min\{1,\Psi_\tau(F)\}$ & $1-\tau(F)$ \\
\midrule
%(body)s
\bottomrule
\end{tabular}
\end{table}
""" % {
            'beta1':    p['beta1'],
            'beta2':    p['beta2'],
            'tau0':     p['tau0'],
            'cap_note': cap_note,
            'body':     body,
        }
    )

# ==============================================================================
# 4. Build Table 2: full funding-implementation table (incl. R_e)
# ==============================================================================
def build_funding_impl_table(p, F_vals):
    Fc = F_crit(p)
    fc_str = f"{Fc:.1f}" if Fc is not None else "\\text{n/a}"

    header = (
        r"""\begin{table}[!htp]
\centering
\caption{Numerical implementation of the funding response at selected funding
levels, using $\beta_1=%(beta1)g$ and $\beta_2=%(beta2)g$. The screening
proportion is capped, $\tau(F)=\min\{1,\Psi_\tau(F)\}$, and the cap binds for
$F\ge F_{\mathrm{crit}}\approx %(fc)s$ million USD. The rate parameters
$\omega,\sigma,\delta,\varepsilon$ are not capped. $R_e$ is computed with the
capped $\tau(F)$.}
\label{tbl:funding_impl}
\begin{tabular}{@{}lccccccccc@{}}
\toprule
 & & $\omega(F)$ & $\Psi_\tau(F)$ & $\tau(F)$ & $\sigma(F)$ & $\delta(F)$
 & $\varepsilon(F)$ & $R_e(F)$ \\
Type & $m(F)$ & (rate) & (uncapped) & (proportion) & (rate) & (rate) & (rate) & \\
\midrule
""" % {'beta1': p['beta1'], 'beta2': p['beta2'], 'fc': fc_str}
    )

    rows = []
    for F in F_vals:
        f = m(F, p)
        rows.append(
            f"$F={F:g}$ & ${f:.3f}$ & ${p['omega0']*f:.4f}$ & "
            f"${psi_tau(F, p):.4f}$ & ${tau(F, p):.4f}$ & "
            f"${p['sigma0']*f:.4f}$ & ${p['delta0']*f:.4f}$ & "
            f"${p['eps0']*f:.4f}$ & ${compute_Re(F, p):.4f}$ \\\\"
        )
    body = "\n".join(rows)

    footer = (
        r"""\bottomrule
\end{tabular}
\end{table}
"""
    )
    return header + body + "\n" + footer

# ==============================================================================
# 5. Write files and print diagnostics
# ==============================================================================
def main():
    tau_tex  = build_tau_cap_table(params, F_levels)
    impl_tex = build_funding_impl_table(params, F_levels)

    with open("tau_cap_table.tex", "w") as fh:
        fh.write(tau_tex)
    with open("funding_impl_table.tex", "w") as fh:
        fh.write(impl_tex)

    print("=" * 78)
    print("LaTeX tables written to:")
    print(f"  {os.path.abspath('tau_cap_table.tex')}")
    print(f"  {os.path.abspath('funding_impl_table.tex')}")
    print("=" * 78)
    print()

    print("----- tau_cap_table.tex -----")
    print(tau_tex)
    print()
    print("----- funding_impl_table.tex -----")
    print(impl_tex)
    print()

    Fc = F_crit(params)
    print("=" * 78)
    print("Diagnostics (capped tau)")
    print("=" * 78)
    print(f"  beta = {params['beta']}, beta1 = {params['beta1']}, "
          f"beta2 = {params['beta2']}, tau0 = {params['tau0']}")
    if Fc is None:
        print("  Cap never binds (denominator <= 0).")
    else:
        print(f"  F_crit (cap binds for F >= F_crit): {Fc:.4f} million USD")
    print()
    print(f"  {'F':>6}  {'m(F)':>8}  {'Psi_tau':>8}  {'tau':>8}  "
          f"{'1 - tau':>8}  {'R_e':>8}")
    print("  " + "-" * 60)
    for F in F_levels:
        print(f"  {F:6.1f}  {m(F, params):8.4f}  {psi_tau(F, params):8.4f}  "
              f"{tau(F, params):8.4f}  {1.0 - tau(F, params):8.4f}  "
              f"{compute_Re(F, params):8.4f}")
    print()

    # Approximate crossing where R_e = 1
    Re_vals = [compute_Re(F, params) for F in F_levels]
    crossings = []
    for i in range(len(F_levels) - 1):
        if (Re_vals[i] - 1.0) * (Re_vals[i + 1] - 1.0) < 0:
            F_cross = F_levels[i] + (F_levels[i + 1] - F_levels[i]) * \
                      (1.0 - Re_vals[i]) / (Re_vals[i + 1] - Re_vals[i])
            crossings.append(F_cross)
    if crossings:
        print(f"  Approximate R_e = 1 crossing: F ~ "
              f"{', '.join(f'{x:.2f}' for x in crossings)} million USD")
    elif Re_vals[0] > 1 and Re_vals[-1] < 1:
        print("  R_e crosses 1 between the first and last F in the grid.")
    else:
        print("  R_e does not cross 1 within the reported grid.")

if __name__ == "__main__":
    main()