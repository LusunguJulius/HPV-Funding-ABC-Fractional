import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, MaxNLocator

# ------------------------------------------------------------
# Model parameters (values from the table, units: 1/year^nu)
# ------------------------------------------------------------
Pi = 186000.0          # recruitment rate
mu = 0.1               # natural death rate
gamma_c = 0.005        # cancer‑induced death rate
omega = 0.4820         # vaccination rate
sigma = 0.9405         # recovery rate of As
delta = 0.0500         # recovery rate of Is
epsilon = 0.0133       # recovery rate of Ic
rho = 0.25             # waning immunity rate of R
tau = 0.9454           # screening proportion
alpha = 0.5013         # progression rate Ia -> As
phi = 0.5926           # progression rate Is -> Ic
theta = 0.1            # vaccine waning rate (also reduces infection risk)
kappa1 = 0.16          # infectivity of As
kappa2 = 0.8383        # infectivity of Is
kappa3 = 0.6427        # infectivity of Ic

# ------------------------------------------------------------
# Analytical functions for R_e and equilibria
# ------------------------------------------------------------
def compute_Re(beta_val):
    d1 = mu + alpha
    d2 = mu + sigma
    d3 = mu + delta + phi
    d4 = mu + gamma_c + epsilon
    term1 = 1.0 / d1
    term2 = (kappa1 * tau * alpha) / (d1 * d2)
    term3 = (kappa2 * (1 - tau) * alpha) / (d1 * d3)
    term4 = (kappa3 * phi * (1 - tau) * alpha) / (d1 * d3 * d4)
    factor = (theta + mu) / (theta + mu + omega)
    Re = beta_val * factor * (term1 + term2 + term3 + term4)
    return Re

def compute_DFE():
    S0 = Pi * (mu + theta) / (mu * (mu + theta + omega))
    V0 = Pi * omega / (mu * (mu + theta + omega))
    return S0, V0

def compute_EE(beta_val):
    Re = compute_Re(beta_val)
    if Re <= 1:
        return None
    d1 = mu + alpha
    d2 = mu + sigma
    d3 = mu + delta + phi
    d4 = mu + gamma_c + epsilon

    m1 = tau * alpha / (sigma + mu)
    m2 = (1 - tau) * alpha / (mu + delta + phi)
    m3 = phi * (1 - tau) * alpha / ((mu + gamma_c + epsilon) * (mu + delta + phi))

    A = (theta + mu + omega) / (theta + mu)
    K = (1 + m1 + m2 + m3 + (sigma * m1 + delta * m2 + epsilon * m3) / (mu + rho))

    num = (Re - 1) * Pi
    den = mu * K * Re + (Re - 1) * gamma_c * m3
    Ia_star = num / den

    S_star = K / (A * (Re - 1)) * Ia_star
    V_star = (omega / (theta + mu)) * S_star
    As_star = m1 * Ia_star
    Is_star = m2 * Ia_star
    Ic_star = m3 * Ia_star
    R_star = (sigma * m1 + delta * m2 + epsilon * m3) / (mu + rho) * Ia_star
    N_star = S_star + V_star + Ia_star + As_star + Is_star + Ic_star + R_star
    return {'S': S_star, 'V': V_star, 'Ia': Ia_star, 'As': As_star,
            'Is': Is_star, 'Ic': Ic_star, 'R': R_star, 'N': N_star}

# ------------------------------------------------------------
# Fractional ABC solver (unchanged except β is an argument)
# ------------------------------------------------------------
def solve_abc_model(beta_val, nu, h=0.01, T=30.0):
    Nsteps = int(T / h)
    time = np.linspace(0, T, Nsteps + 1)

    # Initial conditions
    S0 = 600000.0
    V0 = 350000.0
    Ia0 = 250000.0
    As0 = 240000.0
    Is0 = 220000.0
    Ic0 = 200000.0
    R0 = 0.0

    # Precompute constants
    B_nu = 1 - nu + nu / gamma(nu)
    c1 = (1 - nu) / B_nu
    c2 = nu * (h ** nu) / (B_nu * gamma(nu + 2))

    # Arrays
    S = np.zeros(Nsteps + 1)
    V = np.zeros(Nsteps + 1)
    Ia = np.zeros(Nsteps + 1)
    As = np.zeros(Nsteps + 1)
    Is = np.zeros(Nsteps + 1)
    Ic = np.zeros(Nsteps + 1)
    R = np.zeros(Nsteps + 1)
    Npop = np.zeros(Nsteps + 1)
    lam = np.zeros(Nsteps + 1)
    fS = np.zeros(Nsteps + 1)
    fV = np.zeros(Nsteps + 1)
    fIa = np.zeros(Nsteps + 1)
    fAs = np.zeros(Nsteps + 1)
    fIs = np.zeros(Nsteps + 1)
    fIc = np.zeros(Nsteps + 1)
    fR = np.zeros(Nsteps + 1)

    # Initialise
    S[0], V[0], Ia[0], As[0], Is[0], Ic[0], R[0] = S0, V0, Ia0, As0, Is0, Ic0, R0
    Npop[0] = S0 + V0 + Ia0 + As0 + Is0 + Ic0 + R0
    lam[0] = beta_val / Npop[0] * (Ia0 + kappa1 * As0 + kappa2 * Is0 + kappa3 * Ic0)

    def compute_f(S_val, V_val, Ia_val, As_val, Is_val, Ic_val, R_val, lam_val):
        fS = Pi - (lam_val + mu + omega) * S_val + rho * R_val + theta * V_val
        fV = omega * S_val - (theta + mu) * V_val - theta * lam_val * V_val
        fIa = lam_val * S_val + theta * lam_val * V_val - (mu + alpha) * Ia_val
        fAs = tau * alpha * Ia_val - (sigma + mu) * As_val
        fIs = (1 - tau) * alpha * Ia_val - (mu + delta + phi) * Is_val
        fIc = phi * Is_val - (mu + gamma_c + epsilon) * Ic_val
        fR = sigma * As_val + delta * Is_val + epsilon * Ic_val - (mu + rho) * R_val
        return fS, fV, fIa, fAs, fIs, fIc, fR

    fS[0], fV[0], fIa[0], fAs[0], fIs[0], fIc[0], fR[0] = compute_f(
        S[0], V[0], Ia[0], As[0], Is[0], Ic[0], R[0], lam[0]
    )

    # Time stepping (predictor‑corrector)
    for n in range(Nsteps):
        sumA_S = 0.0; sumA_V = 0.0; sumA_Ia = 0.0; sumA_As = 0.0
        sumA_Is = 0.0; sumA_Ic = 0.0; sumA_R = 0.0
        sumB_S = 0.0; sumB_V = 0.0; sumB_Ia = 0.0; sumB_As = 0.0
        sumB_Is = 0.0; sumB_Ic = 0.0; sumB_R = 0.0

        for k in range(0, n + 1):
            diff = n + 1 - k
            term1 = diff ** nu * (diff + 1 + nu)
            term2 = (n - k) ** nu * (n - k + 2 + 2 * nu) if n - k > 0 else 0.0
            A_coef = term1 - term2

            sumA_S += A_coef * fS[k]
            sumA_V += A_coef * fV[k]
            sumA_Ia += A_coef * fIa[k]
            sumA_As += A_coef * fAs[k]
            sumA_Is += A_coef * fIs[k]
            sumA_Ic += A_coef * fIc[k]
            sumA_R += A_coef * fR[k]

            if k >= 1:
                if n - k == 0:
                    B_coef = diff ** (nu + 1)
                else:
                    B_coef = diff ** (nu + 1) - (n - k) ** nu * (n - k + 1 + nu)
                sumB_S += B_coef * fS[k-1]
                sumB_V += B_coef * fV[k-1]
                sumB_Ia += B_coef * fIa[k-1]
                sumB_As += B_coef * fAs[k-1]
                sumB_Is += B_coef * fIs[k-1]
                sumB_Ic += B_coef * fIc[k-1]
                sumB_R += B_coef * fR[k-1]

        sumA_S *= c2; sumA_V *= c2; sumA_Ia *= c2; sumA_As *= c2
        sumA_Is *= c2; sumA_Ic *= c2; sumA_R *= c2
        sumB_S *= c2; sumB_V *= c2; sumB_Ia *= c2; sumB_As *= c2
        sumB_Is *= c2; sumB_Ic *= c2; sumB_R *= c2

        S_pred = S0 + c1 * fS[n] + sumA_S - sumB_S
        V_pred = V0 + c1 * fV[n] + sumA_V - sumB_V
        Ia_pred = Ia0 + c1 * fIa[n] + sumA_Ia - sumB_Ia
        As_pred = As0 + c1 * fAs[n] + sumA_As - sumB_As
        Is_pred = Is0 + c1 * fIs[n] + sumA_Is - sumB_Is
        Ic_pred = Ic0 + c1 * fIc[n] + sumA_Ic - sumB_Ic
        R_pred = R0 + c1 * fR[n] + sumA_R - sumB_R

        N_pred = S_pred + V_pred + Ia_pred + As_pred + Is_pred + Ic_pred + R_pred
        lam_pred = beta_val / N_pred * (Ia_pred + kappa1 * As_pred + kappa2 * Is_pred + kappa3 * Ic_pred)

        fS_pred, fV_pred, fIa_pred, fAs_pred, fIs_pred, fIc_pred, fR_pred = compute_f(
            S_pred, V_pred, Ia_pred, As_pred, Is_pred, Ic_pred, R_pred, lam_pred
        )

        S[n+1] = S0 + (c1 + c2) * fS_pred + sumA_S - sumB_S
        V[n+1] = V0 + (c1 + c2) * fV_pred + sumA_V - sumB_V
        Ia[n+1] = Ia0 + (c1 + c2) * fIa_pred + sumA_Ia - sumB_Ia
        As[n+1] = As0 + (c1 + c2) * fAs_pred + sumA_As - sumB_As
        Is[n+1] = Is0 + (c1 + c2) * fIs_pred + sumA_Is - sumB_Is
        Ic[n+1] = Ic0 + (c1 + c2) * fIc_pred + sumA_Ic - sumB_Ic
        R[n+1] = R0 + (c1 + c2) * fR_pred + sumA_R - sumB_R

        Npop[n+1] = S[n+1] + V[n+1] + Ia[n+1] + As[n+1] + Is[n+1] + Ic[n+1] + R[n+1]
        lam[n+1] = beta_val / Npop[n+1] * (Ia[n+1] + kappa1 * As[n+1] + kappa2 * Is[n+1] + kappa3 * Ic[n+1])
        fS[n+1], fV[n+1], fIa[n+1], fAs[n+1], fIs[n+1], fIc[n+1], fR[n+1] = compute_f(
            S[n+1], V[n+1], Ia[n+1], As[n+1], Is[n+1], Ic[n+1], R[n+1], lam[n+1]
        )

    return time, S, V, Ia, As, Is, Ic, R

# ------------------------------------------------------------
# Plotting style: axes through origin, thick lines, no grid, x‑axis exactly [0,30]
# ------------------------------------------------------------
def style_axes(ax, T, y_data_lists):
    # Remove top/right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Place left/bottom spines at (0,0)
    ax.spines['left'].set_position(('data', 0))
    ax.spines['bottom'].set_position(('data', 0))
    # Thicken spines
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    # x‑axis exactly up to T
    ax.set_xlim(0, T)
    # y‑axis from 0 to max+5%
    max_y = max([max(y) for y in y_data_lists])
    ax.set_ylim(0, max_y * 1.05)
    ax.grid(False)
    # Tick locators and formatters
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune='both'))
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 3))
    ax.yaxis.set_major_formatter(formatter)
    # Tick and label sizes
    ax.tick_params(axis='both', which='major', labelsize=12, width=2, length=6)
    ax.xaxis.label.set_size(14)
    ax.yaxis.label.set_size(14)
    ax.title.set_size(16)

# ------------------------------------------------------------
# Main: run for two β values, each with all ν, and plot
# ------------------------------------------------------------
beta_list = [0.5546, 2.0]
nu_list = [0.7, 0.8, 0.9, 1.0]
T = 30.0

for beta in beta_list:
    print("\n" + "="*60)
    print(f"β = {beta}")
    print("="*60)

    # Analytical results
    Re = compute_Re(beta)
    print(f"R_e = {Re:.4f}")
    S0, V0 = compute_DFE()
    print(f"Disease‑free equilibrium: S0 = {S0:.2f}, V0 = {V0:.2f}, N0 = {S0+V0:.2f}")
    if Re > 1:
        EE = compute_EE(beta)
        print("Endemic equilibrium (positive):")
        for key, val in EE.items():
            print(f"  {key}* = {val:.2f}")
    else:
        print("No endemic equilibrium (R_e < 1).")

    # Run fractional simulations for all ν
    results = {}
    for nu in nu_list:
        print(f"  Simulating ν = {nu} ...")
        results[nu] = solve_abc_model(beta, nu, h=0.01, T=T)
        print("    done.")

    # Plot compartments for this β
    compartments = [
        ('S', 'Susceptible (S)'),
        ('V', 'Vaccinated (V)'),
        ('Ia', 'Acute infection (I_a)'),
        ('As', 'Screened (A_s)'),
        ('Is', 'Symptomatic (I_s)'),
        ('Ic', 'Cancer (I_c)'),
        ('R', 'Recovered (R)')
    ]

    for var_name, var_label in compartments:
        fig, ax = plt.subplots(figsize=(10, 6))
        y_list = []
        for nu in nu_list:
            time, S, V, Ia, As, Is, Ic, R_arr = results[nu]
            if var_name == 'S':
                y = S
            elif var_name == 'V':
                y = V
            elif var_name == 'Ia':
                y = Ia
            elif var_name == 'As':
                y = As
            elif var_name == 'Is':
                y = Is
            elif var_name == 'Ic':
                y = Ic
            else:
                y = R_arr
            ax.plot(time, y, label=f'ν = {nu}', linewidth=3)
            y_list.append(y)
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('Population')
        ax.set_title(f'{var_label}  (β = {beta})')
        ax.legend(fontsize=12, frameon=False)
        style_axes(ax, T, y_list)
        fig.tight_layout()
        plt.show()

    