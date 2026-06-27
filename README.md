# Blood Inventory Management — Perishable Inventory Simulation

A discrete-event simulation study applying classical inventory theory to perishable blood product management. Models platelet inventory at a hospital over 365 days under stochastic demand and shelf-life constraints, benchmarking a Newsvendor + FEFO policy against a naive ordering baseline.

---

## Problem Statement

Blood platelets have a shelf life of just 5 days, making inventory management unusually difficult. Order too little and you face critical stockouts. Order too much and units expire unused — wastage that is both costly and ethically significant. This project asks:

> *What is the optimal order quantity and issuing policy for a hospital platelet inventory, and how sensitive is system performance to the choice of safety factor?*

---

## Models Implemented

### Newsvendor Model
Single-period optimal order quantity under demand uncertainty.

```
Q* = F⁻¹(CR)       where CR = Cu / (Cu + Co)
```

- `Cu` = underage cost (shortage penalty per unit)
- `Co` = overage cost (holding + wastage cost per unit)
- `CR` = critical ratio — the service level implied by the cost structure
- `Q*` derived analytically as the CR-th quantile of the demand distribution

### Safety Stock & Reorder Point
```
SS  = z_α × σ_L
ROP = μ_L + SS
```

Where `z_α` is the standard normal critical value for target service level α, and `σ_L` is demand standard deviation during lead time.

### FEFO Issuing Policy
First Expired, First Out — units nearest to expiry are issued first, minimising wastage compared to FIFO. WHO-recommended practice for blood and pharmaceutical inventory.

---

## Simulation Design

Built using **SimPy** (discrete-event simulation framework).

| Parameter | Value |
|---|---|
| Simulation horizon | 365 days |
| Product | Platelets |
| Shelf life | 5 days |
| Mean daily demand | 25 units |
| Demand std dev | 7 units |
| Demand distribution | Normal (truncated at 0) |
| Lead time | 1 day (deterministic) |
| Review policy | Continuous (Q,R) |

### SimPy Processes
- **`demand_arrival`** — samples daily demand, issues FEFO, records shortages
- **`replenishment_cycle`** — checks ROP, places orders, receives with lead time
- **`expiry_check`** — daily sweep removing expired units, logging wastage

---

## Key Results

### Policy Comparison

| Policy | Service Level | Wastage Rate | Total Cost |
|---|---|---|---|
| Naive (fixed order = mean) | ~78% | ~22% | High |
| Newsvendor + FEFO | ~95% | ~8% | Optimal |

The Newsvendor + FEFO policy achieves the target service level while substantially reducing wastage compared to naive fixed-quantity ordering.

### Analytical Validation

Simulation output was validated against the closed-form Newsvendor solution across service levels z = 0.5 → 2.0. Simulated fill rates converged to analytical predictions within ±1.5% across all parameter settings, confirming model correctness.

### Sensitivity Analysis — Safety Factor z

One-at-a-time sensitivity on safety factor z across 7 levels:

| z | Service Level | Safety Stock | Wastage Rate |
|---|---|---|---|
| 0.5 | ~69% | 3.5 | ~4% |
| 1.0 | ~84% | 7.0 | ~6% |
| 1.28 | ~90% | 9.0 | ~7% |
| 1.65 | ~95% | 11.6 | ~9% |
| 1.96 | ~97.5% | 13.7 | ~12% |
| 2.33 | ~99% | 16.3 | ~16% |
| 2.58 | ~99.5% | 18.1 | ~19% |

**Key finding:** Moving from z = 1.65 (95% SL) to z = 2.33 (99% SL) doubles the wastage rate for only a 4 percentage point gain in service level — diminishing returns set in sharply above z = 1.65.

---

## Project Structure

```
blood-inventory/
│
├── simulation.py          # SimPy simulation engine
├── newsvendor.py          # Newsvendor model (analytical)
├── fefo_policy.py         # FEFO & FIFO issuing classes
├── sensitivity.py         # Sensitivity analysis runner
├── plots.py               # Visualisation utilities
│
├── notebooks/
│   └── analysis.ipynb     # Full walkthrough and results
│
└── README.md
```

---

## Setup & Usage

```bash
git clone https://github.com/your-username/blood-inventory.git
cd blood-inventory
pip install -r requirements.txt
python simulation.py
```

To run sensitivity analysis:
```bash
python sensitivity.py
```

To open the notebook:
```bash
jupyter notebook notebooks/analysis.ipynb
```

---

## Dependencies

```
simpy==4.1.1
numpy==1.26.4
pandas==2.2.1
scipy==1.13.0
matplotlib==3.8.4
jupyter==1.0.0
```

---

## Theory & References

- **Newsvendor Model:** Silver, E.A., Pyke, D.F., & Peterson, R. (1998). *Inventory Management and Production Planning and Scheduling.* Wiley. Chapter 10.
- **Perishable Inventory:** Nahmias, S. (1975). Optimal ordering policies for perishable inventory. *Operations Research, 23*(4), 735–749.
- **FEFO Policy:** WHO (2009). *Blood cold chain: guidelines for implementation.*
- **SimPy:** Matloff, N. (2008). Introduction to discrete-event simulation.

---

## Author

**Jalli Raja Nandini**
B.Tech. Industrial and Systems Engineering — IIT Kharagpur (2028)
