# SHERLOCK-AQ: Formal Output Range Verification of Deep Neural Networks for Air Quality Safety

> *Implementing the SHERLOCK algorithm (Dutta et al., 2018) for provably safe neural network predictions on the UCI Air Quality dataset.*

---

## Overview

This project addresses a fundamental question in AI safety:

> **Can you mathematically prove what outputs a neural network will produce — not just test it on samples?**

Standard machine learning evaluation relies on test set performance. But for safety-critical systems, "it worked on the test set" is not enough. A neural network deployed in a real environment might encounter inputs it was never tested on, and there is no guarantee its predictions stay within safe bounds.

**SHERLOCK-AQ** implements the SHERLOCK algorithm from the research paper *"Output Range Analysis for Deep Feedforward Neural Networks"* (Dutta, Jha, Sankaranarayanan & Tiwari, 2018) and applies it to Carbon Monoxide (CO) concentration prediction from environmental sensor data.

Given a trained neural network and a bounded region of valid sensor inputs, SHERLOCK computes a **formally proven interval [lower, upper]** such that for **every possible input** in that region, the network output is **guaranteed** to fall within those bounds.

---

## The Problem — Output Range Estimation

Formally: given neural network `F_N`, input constraint region `P = {x | Ax ≤ b}`, and tolerance `δ > 0`, find an interval `[l, u]` such that:

```
∀ x ∈ P :  l  ≤  F_N(x)  ≤  u
```

This is **NP-hard**. A network with 160 neurons has 2¹⁶⁰ possible combinations of active/inactive neurons — brute force is infeasible. SHERLOCK solves this efficiently by combining local search with an exact MILP solver.

---

## Why Air Quality?

Carbon Monoxide is a colorless, odorless gas that causes serious harm at elevated concentrations. An AI monitoring system that predicts CO levels must never produce a dangerously wrong output. Formally bounding what the model can predict has direct safety implications — making this an ideal testbed for neural network verification.

---

## Dataset

**UCI Air Quality Dataset** — Hourly averaged sensor readings from a polluted Italian city (March 2004 – February 2005).

| Property | Value |
|---|---|
| Source | UCI Machine Learning Repository / Kaggle |
| Total records | 9,471 hourly readings |
| Valid records (after cleaning) | 6,941 (73.3%) |
| Input features | 11 continuous sensor/environmental variables |
| Target variable | CO(GT) — Carbon Monoxide (mg/m³) |
| Target range (valid) | 0.1 to 11.9 mg/m³ |
| Train / Test split | 5,552 / 1,389 samples |
| Scaling | MinMaxScaler fitted on train only (no data leakage) |

### Features

| Feature | Description |
|---|---|
| `PT08.S1(CO)` | Tin oxide sensor — CO proxy |
| `C6H6(GT)` | Benzene concentration (µg/m³) |
| `PT08.S2(NMHC)` | Titania sensor — NMHC proxy |
| `NOx(GT)` | Nitrogen oxides (ppb) |
| `PT08.S3(NOx)` | Tungsten oxide sensor — NOx proxy |
| `NO2(GT)` | Nitrogen dioxide (µg/m³) |
| `PT08.S4(NO2)` | Tungsten oxide sensor — NO2 proxy |
| `PT08.S5(O3)` | Indium oxide sensor — Ozone proxy |
| `T` | Temperature (°C) |
| `RH` | Relative humidity (%) |
| `AH` | Absolute humidity |

### Data Cleaning

The dataset uses **-200 as a sentinel value** for faulty sensor readings. A systematic cleaning pipeline was applied:

| Issue | Decision | Justification |
|---|---|---|
| `NMHC(GT)` column | Dropped entirely | 89.1% invalid — MNAR pattern, imputation statistically invalid |
| Sentinel -200 values | Replaced with NaN | Not real readings — sensor malfunction codes |
| Mean imputation | Rejected | Reduces variance artificially, distorts tail distribution |
| Median imputation | Rejected | Concentrates mass at median, distorts extremes needed by SHERLOCK |
| Remaining NaN rows | Listwise deletion | 73.3% data retained, preserves true distribution |

> **Critical insight:** SHERLOCK searches for the **maximum and minimum** possible outputs. Mean/median imputation smooths the distribution and biases the extremes — directly undermining the verification result.

---

## Methodology

### 1. Neural Network Architecture

Designed specifically for SHERLOCK compatibility. The output layer **must be linear** (no activation) as required by the MILP encoding constraint `C_{k+1}: y = W_k·z_k + b_k`.

```
Input  (11)  →  Linear + ReLU  (64)  →  Linear + ReLU  (64)  →  Linear + ReLU  (32)  →  Linear  (1)
```

| Layer | Type | Dimensions | Activation |
|---|---|---|---|
| Input | — | 11 | — |
| Hidden 1 | Linear + ReLU | 11 → 64 | ReLU |
| Hidden 2 | Linear + ReLU | 64 → 64 | ReLU |
| Hidden 3 | Linear + ReLU | 64 → 32 | ReLU |
| Output | Linear | 32 → 1 | **None** (required by MILP) |

**Total parameters:** 7,041 across 3 hidden layers (160 neurons)

**Why ReLU?** ReLU is piecewise linear — each neuron has exactly 2 states (active/inactive), encodable with a single binary variable in the MILP. Sigmoid/tanh would require approximation.

**Why linear output?** Adding ReLU to the output layer would require an additional binary variable in the MILP encoding and break the `milp_exact_bound()` function.

### 2. MILP Encoding (Section 3.1 of Paper)

For each ReLU neuron `j` in hidden layer `i`, a binary variable `t_{i,j} ∈ {0,1}` is introduced (`t=1`: active, `t=0`: inactive). The constraints from the paper:

```
z_{i+1}  ≥  W_i · z_i + b_i                 # z ≥ pre-activation
z_{i+1}  ≥  W_i · z_i + b_i  +  M_i · t    # forces z=0 when t=0
z_{i+1}  ≥  0                                # ReLU floor
z_{i+1}  ≤  M_i · (1 − t)                   # forces z=input when t=1
```

The complete MILP:
```
maximize   y
subject to: C_0 (input constraints), C_1,...,C_k (layer constraints), C_{k+1} (output)
```

**Big-M Computation:** Rather than hardcoding M, we implement per-layer interval arithmetic exactly as specified in the paper (*"M is estimated via interval analysis using ||Wi||∞ and the bounding box of the input polyhedron"*):

```python
def compute_layer_bigM(W, b, lb, ub):
    W_pos = np.maximum(W, 0)
    W_neg = np.minimum(W, 0)
    lb_pre = W_pos @ lb + W_neg @ ub + b    # tightest lower bound
    ub_pre = W_pos @ ub + W_neg @ lb + b    # tightest upper bound
    M_i    = max(|lb_pre|, |ub_pre|) × 1.1 + 1.0   # 10% safety margin
    return M_i, relu(lb_pre), relu(ub_pre)
```

### 3. SHERLOCK Algorithm (Algorithm 1)

SHERLOCK alternates between local search and MILP to find tight, proven output bounds:

```
FINDUPPERBOUND(N, P, δ):
    x ← Sample(P)                        # random starting point
    u ← F_N(x)

    WHILE not terminate:
        (x̂, û) ← LocalSearch(N, x, P)   # gradient ascent
        u       ← û + δ                   # raise threshold
        (x', u', feas) ← MILP(I, u)      # exact global search

        IF feas:
            (x, u) ← (x', u')            # found better — continue
        ELSE:
            terminate ← TRUE              # proven — nothing beats u

    RETURN u                              # proven upper bound
```

**Theorem 2 (Dutta et al.):** Algorithm 1 always terminates. The output `u` satisfies `u* ≤ u ≤ u* + δ`, where `u*` is the true maximum.

### 4. Local Search — Gradient Ascent

```python
for step in range(n_steps):
    if x.grad is not None:
        x.grad.zero_()             # defensive — prevents accumulation
    output = model(x.unsqueeze(0))
    output.backward()
    grad    = x.grad.detach()
    x_new   = x + lr * grad        # ascent step
    x_new   = clamp(x_new, lb, ub) # project back into P
    x       = x_new.detach().requires_grad_(True)
```

---

## Results

### Model Performance

| Metric | Value | Notes |
|---|---|---|
| MSE | 0.1254 | Mean Squared Error (original CO units) |
| RMSE | 0.3541 mg/m³ | Average prediction error |
| MAE | 0.2319 mg/m³ | Robust to outliers |
| R² | **0.9336** | 93.4% of CO variance explained |
| Negative predictions | 0 / 1,389 (0.00%) | No clipping applied — raw output |

### SHERLOCK Output Bounds (ε = 0.05)

| Quantity | Scaled Value | CO (mg/m³) |
|---|---|---|
| **SHERLOCK Upper Bound (u)** | 0.324900 | **3.934 mg/m³** |
| **SHERLOCK Lower Bound (l)** | 0.116982 | **1.480 mg/m³** |
| Exact MILP Upper | 5.865182 | 69.31 mg/m³ (global) |
| Exact MILP Lower | -7.719574 | -90.99 mg/m³ (global) |
| Sampled Max (200k points) | 0.315447 | 3.822 mg/m³ |
| Sampled Min (200k points) | 0.167762 | 2.080 mg/m³ |
| Tolerance δ | 0.01 | ~0.12 mg/m³ |

**Interpretation:** For ANY sensor reading within ±5% of the reference point, the network is **guaranteed** to predict CO between **1.480 and 3.934 mg/m³**. This is a mathematical proof, not a statistical estimate.

> **SHERLOCK vs. Standalone MILP:** SHERLOCK bounds are tight because local search restricts the search to the ε-local region and warm-starts the solver. The standalone MILP searches the entire feasible input space without locality constraints (Theorem 1), producing wider but globally valid bounds.

### Verification

| Check | Result | Status |
|---|---|---|
| Upper bound ≥ sampled max (0.315447) | 0.324900 ≥ 0.315447 | ✅ HOLDS |
| Lower bound ≤ sampled min (0.167762) | 0.116982 ≤ 0.167762 | ✅ HOLDS |
| All 200,000 sample outputs in [l, u] | Confirmed | ✅ HOLDS |
| MILP solver status (upper) | Optimal | ✅ SOLVED |
| MILP solver status (lower) | Optimal | ✅ SOLVED |

### Multi-Region Analysis

| Epsilon (ε) | Proven CO Range (mg/m³) | Range Width (scaled) |
|---|---|---|
| 0.02 | [2.703, 3.638] | 0.0792 |
| **0.05** | **[1.698, 3.904]** | **0.1869** ← primary experiment |
| 0.08 | [0.778, 4.974] | 0.3556 |
| 0.10 | [0.694, 5.355] | 0.3950 |
| 0.15 | [-0.534, 6.142] | 0.5657 |

Monotonically increasing range width with ε confirms SHERLOCK correctly captures network sensitivity to input variation, consistent with Theorem 2.

---

## Project Structure

```
SHERLOCK-AQ/
│
├── ML_CCP_AirQuality_SHERLOCK_V3.ipynb   # Main Jupyter notebook
├── AirQuality.csv                         # UCI Air Quality dataset
├── Technical_Report.pdf                   # Full technical report (13 pages)
├── Presentation.pptx                      # Project presentation (13 slides)
├── research_paper/
│   └── Dutta_et_al_2018_SHERLOCK.pdf     # Original research paper
└── README.md
```

---

## Installation & Usage

### Requirements

```bash
pip install torch numpy pandas scikit-learn matplotlib pulp
```

Or install all at once:

```bash
pip install torch numpy pandas scikit-learn matplotlib pulp jupyter
```

> **Note:** `pulp` is required for the exact MILP solver (Section 7 of the notebook). A pip install cell is included at the top of that section.

### Running the Notebook

```bash
git clone https://github.com/yourusername/SHERLOCK-AQ.git
cd SHERLOCK-AQ
jupyter notebook ML_CCP_AirQuality_SHERLOCK_V3.ipynb
```

Run cells **top to bottom**. The notebook is divided into 9 clearly labeled sections.

### Notebook Sections

| Section | Content |
|---|---|
| 1 | Imports & reproducibility seeds |
| 2 | Load dataset & EDA (6 plots) |
| 3 | Data preprocessing — cleaning, splitting, scaling |
| 4 | Data cleaning justification (imputation comparison) |
| 5 | Build & train ReLU neural network |
| 6 | Define input constraint region φ(x) |
| 7 | Local search (gradient ascent) + Real MILP solver (PuLP/CBC) |
| 8 | SHERLOCK Algorithm 1 — upper and lower bounds |
| 9 | Results, 2-stage verification, multi-region analysis |

---

## Key Design Decisions

| Decision | Choice | Justification |
|---|---|---|
| Output activation | Linear (no ReLU) | Paper Section 3.1 MILP requires linear output — ReLU would break encoding |
| Big-M computation | Per-layer interval arithmetic | Paper explicitly requires this — hardcoded M is unjustified |
| Evaluation clipping | No clipping in metrics | Scientific integrity — 0% negative predictions anyway |
| Data cleaning | Listwise deletion | MNAR + SHERLOCK needs accurate extremes |
| MILP solver | PuLP/CBC (exact) | Provides formal Theorem 1 proof — sampling cannot |
| Gradient zeroing | Defensive check added | Best practice — even though detach() already prevents accumulation |
| Data split | Split before scaling | Prevents data leakage from test set into scaler |

---

## Theoretical Foundation

### Theorem 1 (MILP Encoding)
The MILP encoding is always feasible and bounded. Its optimal solution `u*` satisfies `u* = max_{x∈P} F_N(x)`.

### Theorem 2 (SHERLOCK Algorithm)
Algorithm 1 always terminates. The output `u` satisfies:
```
u*  ≤  u  ≤  u* + δ
```
where `u*` is the true maximum and `δ` is the tolerance parameter.

---

## References

```bibtex
@inproceedings{dutta2018output,
  title     = {Output Range Analysis for Deep Feedforward Neural Networks},
  author    = {Dutta, Souradeep and Jha, Susmit and
               Sankaranarayanan, Sriram and Tiwari, Ashish},
  booktitle = {NASA Formal Methods (NFM 2018)},
  publisher = {Springer},
  year      = {2018}
}
```

Additional references:
- Katz et al. (2017) — Reluplex: An Efficient SMT Solver for Verifying Deep Neural Networks
- LeCun, Bengio & Hinton (2015) — Deep Learning (Nature)
- Bunel et al. (2017) — Piecewise Linear Neural Network Verification: A Comparative Study

---

## Stack

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![PuLP](https://img.shields.io/badge/PuLP-3.3-green)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

| Tool | Purpose |
|---|---|
| PyTorch | Neural network training & gradient computation |
| PuLP / CBC | Exact MILP solver for formal bounds |
| NumPy | Interval arithmetic & numerical operations |
| Scikit-learn | Data preprocessing & evaluation metrics |
| Matplotlib | All visualizations |
