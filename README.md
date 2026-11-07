# Healthcare Funding in an ABC Fractional-Order HPV Model — Python Code

Python code accompanying the manuscript:

> **The Role of Healthcare Funding in HPV Transmission and Cervical Cancer Control: An Atangana-Baleanu-Caputo Fractional-Order Approach**
> Lusungu Julius Mbigili, Nkuba Nyerere, Aziza Iddi and Saul Mpeshe

The model is an HPV / cervical cancer transmission model with the Atangana-Baleanu-Caputo (ABC) fractional derivative, in which healthcare funding *F* (million USD) modifies vaccination, screening and treatment rates. Numerical solutions use an Adams-Bashforth-Moulton (ABM) type scheme for the ABC operator.

## Contents

### `HPV_Funding_Model/`
| Script | What it does |
|---|---|
| `Simulation_ABC_Fractional_HPV_Funding_Corrected.py` | Main simulation of the funding model: compartment dynamics under different funding levels, total infected, effective reproduction number R_e (bar chart and contour), and solver convergence. |
| `generate_funding_tables.py` | Self-contained script for the funding response with the screening cap, tau(F) = min{1, tau0·m(F)}; computes R_e(F) and writes LaTeX tables `tau_cap_table.tex` and `funding_impl_table.tex`. |

### `Fractional_HPVmodel/`
| Script | What it does |
|---|---|
| `ABC_ABM_Scheme_Calibrated_Params2.py` | Solves the ABC fractional model with the calibrated parameters and plots all state variables for fractional orders ν = 0.7, 0.8, 0.9, 1.0 and different values of β (figures are shown on screen, not saved). |
| `ABC_ABM_Scheme_Calibratedparams_FundingFunc1.py` | Transient analysis across fractional orders with the funding function (bold-style figures); writes CSV results, figures and LaTeX tables. |
| `transient_analysis_corrected.py` | Corrected quantitative transient analysis: decay rates, eigenvalue/Matignon stability check, step-size (h) refinement; writes CSV results, figures and LaTeX tables. |
| `ABC_with_Funding_Sensitivity_Analysis2.py` | Global sensitivity analysis (Latin hypercube sampling + PRCC) of the funding model; writes `sensitivity_results_funding.csv`, `prcc_table_funding.tex` and `ImagesF/prcc_barchart_funding.png`. |
| `ABC_fractional_hpv_funding_synthetic_dataset_simulation1.py` | Calibration, validation and identifiability: synthetic datasets, least-squares fit, confidence intervals, profile likelihoods and LaTeX tables. |

## Requirements

Python 3.10 or newer (developed with Python 3.12). Install the packages with:

```bash
pip install -r requirements.txt
```

`joblib` is only used to run the bootstrap / profile-likelihood computations in parallel.

## How to run

Run each script from inside its folder, for example:

```bash
cd HPV_Funding_Model
python Simulation_ABC_Fractional_HPV_Funding_Corrected.py
```

Figures (`.png`), data (`.csv`) and LaTeX tables (`.tex`) are written to the current folder.

The number of Latin hypercube samples in the sensitivity analysis can be changed with an environment variable (default 1000):

```bash
N_SAMPLES=500 python ABC_with_Funding_Sensitivity_Analysis2.py
```

## Citation

If you use this code, please cite the manuscript above and this software archive (DOI on Zenodo).

## License

MIT License — see `LICENSE`.

## Contact

Lusungu Julius Mbigili — Department of Mathematics and Statistics Studies, Mzumbe University, Tanzania
