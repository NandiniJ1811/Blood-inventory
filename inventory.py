"""
models/inventory.py
===================
Classical inventory models extended for cold chain perishable products.

Theoretical backbone:
  - Newsvendor Model: Nahmias (1975), "One-Period Models with Stochastic Demand"
  - EOQ: Harris (1913), Wilson (1934) — minimise total ordering + holding cost
  - (Q,R) Continuous Review: Silver, Pyke & Peterson (1998), "Inventory Management
    and Production Planning and Scheduling", 3rd ed.
  - FEFO Policy: WHO (2019) Good Distribution Practice for blood products

The core extension in this project: classical models assume demand is the only
uncertainty. Cold chain adds temperature uncertainty — batches may be partially or
fully lost to excursions. We adjust Q* and R upward to account for effective yield
being stochastic, not just demand.

Author: IIT Kharagpur, Dept. of Industrial and Systems Engineering
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, optimize
from typing import Optional, List, Tuple, Dict
import warnings

# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# 1. NEWSVENDOR MODEL
# ─────────────────────────────────────────────────────────────────────────────

class NewsvendorModel:
    """
    Single-period inventory model under demand uncertainty.

    The classic Newsvendor (or Newsboy) problem:
        min  E[Co * max(Q - D, 0) + Cu * max(D - Q, 0)]
    where:
        Co = overage cost (holding unsold stock) = h (holding cost)
        Cu = underage cost (shortage penalty) = p (shortage/lost-sale cost)

    Optimal solution:
        Q* = F^{-1}(CR)   where CR = Cu / (Cu + Co)  ... (1)
    F^{-1} is the quantile function (inverse CDF) of demand.

    Reference: Nahmias, S. (1975). "One-Period Models with Stochastic Demand."
               Chapter in: Production and Operations Analysis.
    """

    def __init__(
        self,
        holding_cost: float,
        shortage_cost: float,
        unit_cost: float,
        seed: int = DEFAULT_SEED,
    ):
        """
        Parameters
        ----------
        holding_cost  : Co — cost per unit per period for excess inventory
        shortage_cost : Cu — cost per unit short (penalty + lost goodwill)
        unit_cost     : c  — purchase/manufacturing cost per unit
        seed          : random seed for reproducibility
        """
        if holding_cost < 0 or shortage_cost < 0 or unit_cost < 0:
            raise ValueError("All costs must be non-negative.")
        self.Co = holding_cost      # overage cost
        self.Cu = shortage_cost     # underage cost
        self.c  = unit_cost
        self.rng = np.random.default_rng(seed)

    # ── Critical Ratio ────────────────────────────────────────────────────────

    def critical_ratio(self) -> float:
        """
        Compute the critical ratio (service level target implied by costs).

            CR = Cu / (Cu + Co)   ... equation (1), Nahmias (1975)

        At Q*, the probability of demand NOT exceeding Q equals CR.
        Interpretation: if CR = 0.85, we stock at the 85th percentile of demand.

        Returns
        -------
        float in (0, 1)
        """
        # CR = Cu / (Cu + Co) — Newsvendor critical ratio
        cr = self.Cu / (self.Cu + self.Co)
        return cr

    # ── Optimal Order Quantity ────────────────────────────────────────────────

    def optimal_order_quantity(
        self,
        demand_mean: float,
        demand_std: float,
        distribution: str = "normal",
    ) -> float:
        """
        Compute Q* = F^{-1}(CR) — optimal order quantity.

        Supports Normal and Poisson demand distributions.

        Parameters
        ----------
        demand_mean  : μ — expected demand over the period
        demand_std   : σ — standard deviation of demand
        distribution : 'normal' or 'poisson'

        Returns
        -------
        Q* (float) — optimal stocking quantity
        """
        cr = self.critical_ratio()

        if distribution == "normal":
            # Q* = μ + z_CR × σ, where z_CR = Φ^{-1}(CR) — standard normal quantile
            # Reference: Silver et al. (1998), eq. 11.9
            Q_star = stats.norm.ppf(cr, loc=demand_mean, scale=demand_std)

        elif distribution == "poisson":
            # For integer-valued Poisson demand, Q* is the smallest integer q
            # such that F(q) >= CR  (discrete analogue of eq. 1)
            from scipy.stats import poisson
            q = 0
            while poisson.cdf(q, mu=demand_mean) < cr:
                q += 1
            Q_star = float(q)

        else:
            raise ValueError(f"Unsupported distribution: {distribution}")

        return max(0.0, Q_star)

    # ── Expected Cost ─────────────────────────────────────────────────────────

    def expected_cost(
        self,
        Q: float,
        demand_mean: float,
        demand_std: float,
    ) -> Dict[str, float]:
        """
        Expected total cost at order quantity Q under Normal demand.

        E[Cost(Q)] = Co × E[max(Q-D,0)] + Cu × E[max(D-Q,0)]
                   = Co × (Q-μ)Φ(z) + Co × σ φ(z)
                     + Cu × σ [φ(z) - z(1-Φ(z))]
        where z = (Q - μ) / σ,  φ = standard normal PDF, Φ = CDF.

        Reference: Silver et al. (1998), eq. 11.6–11.8

        Returns
        -------
        dict with keys: 'holding', 'shortage', 'total'
        """
        mu, sigma = demand_mean, demand_std
        z = (Q - mu) / sigma   # standardised z-score

        phi_z  = stats.norm.pdf(z)     # φ(z) — standard normal PDF
        Phi_z  = stats.norm.cdf(z)     # Φ(z) — standard normal CDF

        # Expected overage: E[max(Q-D,0)] = (Q-μ)Φ(z) + σφ(z)
        expected_overage  = (Q - mu) * Phi_z + sigma * phi_z
        # Expected underage (loss function): E[max(D-Q,0)] = σ[φ(z) - z(1-Φ(z))]
        expected_underage = sigma * (phi_z - z * (1 - Phi_z))

        holding_cost  = self.Co * expected_overage
        shortage_cost = self.Cu * expected_underage
        total_cost    = holding_cost + shortage_cost

        return {
            "holding":  round(holding_cost, 4),
            "shortage": round(shortage_cost, 4),
            "total":    round(total_cost, 4),
        }

    # ── Perishability Extension ───────────────────────────────────────────────

    def optimal_order_quantity_perishable(
        self,
        demand_mean: float,
        demand_std: float,
        shelf_life_days: int,
        daily_spoilage_rate: float,
    ) -> Dict[str, float]:
        """
        Extended Newsvendor for perishable products (Nahmias, 1975).

        Perishability reduces the effective yield per unit ordered.
        If a unit expires with probability p_expire = f(shelf_life, demand_rate),
        the effective overage cost increases because unsold stock is not just
        held — it perishes and becomes a dead loss.

        Adjusted overage cost:
            Co_eff = Co + c × p_expire
        where c is unit cost and p_expire approximates the spoilage probability.

        Spoilage probability approximation (single-period):
            p_expire ≈ 1 - (1 - daily_spoilage_rate)^shelf_life_days
        Reference: Nahmias (1975), "Perishable Inventory Theory: A Review",
                   Operations Research, 30(4), 680-708.

        Parameters
        ----------
        demand_mean        : μ
        demand_std         : σ
        shelf_life_days    : m — product shelf life in days
        daily_spoilage_rate: r — probability of spoilage per unit per day if unsold

        Returns
        -------
        dict with keys: 'Q_star_perishable', 'Q_star_base', 'critical_ratio_adjusted',
                        'p_expire', 'Co_effective'
        """
        # Probability a unit expires if held full shelf life
        # p_expire = 1 - (1-r)^m
        p_expire = 1.0 - (1.0 - daily_spoilage_rate) ** shelf_life_days

        # Adjusted overage cost: holding PLUS expected dead-loss from spoilage
        Co_eff = self.Co + self.c * p_expire

        # Adjusted critical ratio with effective overage cost
        # CR_adj = Cu / (Cu + Co_eff) — same formula, higher Co → lower CR → lower Q*
        cr_adj = self.Cu / (self.Cu + Co_eff)

        # Q* under perishability
        z_adj  = stats.norm.ppf(cr_adj)
        Q_star_perish = demand_mean + z_adj * demand_std
        Q_star_base   = self.optimal_order_quantity(demand_mean, demand_std)

        return {
            "Q_star_perishable":     round(max(0.0, Q_star_perish), 2),
            "Q_star_base":           round(Q_star_base, 2),
            "critical_ratio_adjusted": round(cr_adj, 4),
            "p_expire":              round(p_expire, 4),
            "Co_effective":          round(Co_eff, 4),
            "Q_reduction_pct":       round(100 * (Q_star_base - max(0, Q_star_perish)) / Q_star_base, 2),
        }

    # ── Plotting utility ──────────────────────────────────────────────────────

    def cost_curve(
        self,
        demand_mean: float,
        demand_std: float,
        Q_range: Optional[Tuple[float, float]] = None,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """
        Plot expected total cost as a function of Q, highlighting Q*.

        Returns matplotlib Figure.
        """
        Q_star = self.optimal_order_quantity(demand_mean, demand_std)
        lo = Q_range[0] if Q_range else max(0, demand_mean - 3 * demand_std)
        hi = Q_range[1] if Q_range else demand_mean + 3 * demand_std
        Q_vals = np.linspace(lo, hi, 300)
        costs  = [self.expected_cost(q, demand_mean, demand_std)["total"] for q in Q_vals]

        fig, ax_ = (plt.subplots(figsize=(8, 4)) if ax is None else (ax.get_figure(), ax))
        ax_ = ax_ if ax is not None else fig.axes[0]
        ax_.plot(Q_vals, costs, color="#2c7bb6", lw=2, label="E[Total Cost]")
        ax_.axvline(Q_star, color="#d7191c", lw=1.5, linestyle="--",
                    label=f"Q* = {Q_star:.1f} (CR={self.critical_ratio():.2f})")
        ax_.set_xlabel("Order Quantity Q")
        ax_.set_ylabel("Expected Cost")
        ax_.set_title("Newsvendor Cost Curve")
        ax_.legend()
        ax_.grid(alpha=0.3)
        plt.tight_layout()
        return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. EOQ MODEL
# ─────────────────────────────────────────────────────────────────────────────

class EOQModel:
    """
    Economic Order Quantity (EOQ) model.

    Minimises total annual cost = ordering cost + holding cost:
        TC(Q) = (D/Q) × K + (Q/2) × H
    Optimal solution:
        Q* = sqrt(2DK/H)   ... Harris (1913), also known as Wilson's formula

    Assumptions: constant deterministic demand, instantaneous replenishment,
    no shortages. These are relaxed in (Q,R) and simulation layers.

    Reference: Harris, F.W. (1913). "How Many Parts to Make at Once."
               Factory, The Magazine of Management, 10(2), 135-136.
    """

    def __init__(
        self,
        demand_rate: float,
        ordering_cost: float,
        holding_cost_rate: float,
        unit_cost: float,
    ):
        """
        Parameters
        ----------
        demand_rate       : D — annual demand (units/year)
        ordering_cost     : K — fixed cost per order placed (₹ or $)
        holding_cost_rate : I — fraction of unit cost held per year (e.g. 0.20)
        unit_cost         : c — purchase/manufacturing cost per unit
        """
        self.D  = demand_rate
        self.K  = ordering_cost
        self.I  = holding_cost_rate
        self.c  = unit_cost
        # Annual holding cost per unit
        self.H  = holding_cost_rate * unit_cost   # H = I × c

    def eoq(self) -> float:
        """
        Classic EOQ formula.

            Q* = sqrt(2 × D × K / H)   ... Harris (1913)

        Returns
        -------
        float — optimal order quantity in units
        """
        # Q* = sqrt(2DS/H) — Economic Order Quantity
        Q_star = np.sqrt(2 * self.D * self.K / self.H)
        return round(Q_star, 2)

    def total_cost(self, Q: float) -> Dict[str, float]:
        """
        Total annual cost at order quantity Q.

            TC(Q) = (D/Q) × K + (Q/2) × H

        Parameters
        ----------
        Q : order quantity (units)

        Returns
        -------
        dict: {'ordering', 'holding', 'purchase', 'total'}
        """
        if Q <= 0:
            raise ValueError("Order quantity Q must be positive.")
        ordering_cost = (self.D / Q) * self.K          # (D/Q) × K
        holding_cost  = (Q / 2) * self.H               # (Q/2) × H
        purchase_cost = self.D * self.c                 # D × c (independent of Q)
        total         = ordering_cost + holding_cost + purchase_cost

        return {
            "ordering": round(ordering_cost, 2),
            "holding":  round(holding_cost, 2),
            "purchase": round(purchase_cost, 2),
            "total":    round(total, 2),
        }

    def sensitivity_plot(
        self,
        Q_range_factor: float = 3.0,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """
        Plot total annual cost vs Q, highlighting Q* at the minimum.

        The classic U-shaped EOQ cost curve: ordering cost decreases as Q
        increases (fewer orders placed), holding cost increases linearly.
        Their sum is minimised at Q* — the EOQ.

        Parameters
        ----------
        Q_range_factor : how far around Q* to plot (±factor × Q*)
        ax             : optional existing Axes

        Returns
        -------
        matplotlib Figure
        """
        Q_star = self.eoq()
        Q_lo   = max(1, Q_star / Q_range_factor)
        Q_hi   = Q_star * Q_range_factor
        Q_vals = np.linspace(Q_lo, Q_hi, 400)

        tc  = [self.total_cost(q)["total"]    for q in Q_vals]
        oc  = [self.total_cost(q)["ordering"] for q in Q_vals]
        hc  = [self.total_cost(q)["holding"]  for q in Q_vals]

        fig, ax_ = (plt.subplots(figsize=(9, 5)) if ax is None else (ax.get_figure(), ax))
        ax_ = fig.axes[0] if ax is None else ax

        ax_.plot(Q_vals, tc,  lw=2.5, color="#2c7bb6", label="Total Cost TC(Q)")
        ax_.plot(Q_vals, oc,  lw=1.5, color="#d7191c", linestyle="--", label="Ordering Cost")
        ax_.plot(Q_vals, hc,  lw=1.5, color="#1a9641", linestyle="--", label="Holding Cost")
        ax_.axvline(Q_star, color="black", lw=1.5, linestyle=":",
                    label=f"EOQ Q* = {Q_star:.0f}")
        ax_.scatter([Q_star], [self.total_cost(Q_star)["total"]],
                    color="black", zorder=5, s=80)
        ax_.set_xlabel("Order Quantity Q (units)")
        ax_.set_ylabel("Annual Cost (₹)")
        ax_.set_title("EOQ Cost Curve — Holding vs Ordering Cost Trade-off\n"
                      "Q* = √(2DS/H) [Harris 1913]")
        ax_.legend()
        ax_.grid(alpha=0.3)
        plt.tight_layout()
        return fig

    def reorder_cycles_per_year(self) -> float:
        """Number of orders placed per year at EOQ: D / Q*"""
        return round(self.D / self.eoq(), 2)

    def cycle_length_days(self) -> float:
        """Average days between orders: 365 / (D/Q*)"""
        return round(365 / self.reorder_cycles_per_year(), 1)


# ─────────────────────────────────────────────────────────────────────────────
# 3. CONTINUOUS REVIEW (Q, R) POLICY
# ─────────────────────────────────────────────────────────────────────────────

class ContinuousReviewQR:
    """
    Continuous Review (Q, R) Inventory Policy.

    Policy: monitor inventory continuously; when inventory position drops to
    reorder point R, place an order of fixed size Q.

    Key formulas:
        Lead time demand: μ_L = μ_d × L
        Lead time demand std: σ_L = sqrt(σ_d² × L + μ_d² × σ_L²)
        Safety stock: SS = z_α × σ_L   ... Silver et al. (1998), eq. 11.23
        Reorder point: R = μ_L + SS    ... Silver et al. (1998), eq. 11.24
        Order quantity: Q = EOQ

    Cold chain extension: if P(excursion) > 0, effective yield per order is
    reduced by spoilage probability. We inflate R to compensate.

    Reference: Silver, E.A., Pyke, D.F., Peterson, R. (1998).
               "Inventory Management and Production Planning and Scheduling",
               3rd ed. John Wiley & Sons. Chapter 11.
    """

    def __init__(
        self,
        demand_mean: float,
        demand_std: float,
        lead_time_mean: float,
        lead_time_std: float,
        holding_cost: float,
        shortage_cost: float,
        ordering_cost: float,
        unit_cost: float = 1.0,
        review_period: float = 0,
    ):
        """
        Parameters
        ----------
        demand_mean   : μ_d — mean daily demand (units/day)
        demand_std    : σ_d — std of daily demand
        lead_time_mean: L̄  — mean replenishment lead time (days)
        lead_time_std : σ_L — std of lead time (days)
        holding_cost  : h   — holding cost per unit per day
        shortage_cost : p   — shortage penalty per unit short
        ordering_cost : K   — fixed cost per order
        unit_cost     : c   — unit purchase cost
        review_period : T   — for periodic review; 0 = continuous review
        """
        self.mu_d  = demand_mean
        self.sig_d = demand_std
        self.L_bar = lead_time_mean
        self.sig_L = lead_time_std
        self.h     = holding_cost
        self.p     = shortage_cost
        self.K     = ordering_cost
        self.c     = unit_cost
        self.T     = review_period

    # ── Lead time demand moments ──────────────────────────────────────────────

    def lead_time_demand_moments(self) -> Tuple[float, float]:
        """
        Compute mean and std of demand over lead time.

        When both demand and lead time are stochastic:
            μ_L = μ_d × L̄
            σ_L = sqrt(L̄ × σ_d² + μ_d² × σ_L²)   ... Silver et al. eq. 11.21

        Returns (μ_L, σ_L)
        """
        mu_L  = self.mu_d * self.L_bar
        # Compound variance: demand variance × lead time + demand mean² × lead time var
        sig_L = np.sqrt(self.L_bar * self.sig_d**2 + self.mu_d**2 * self.sig_L**2)
        return mu_L, sig_L

    # ── Safety Stock ──────────────────────────────────────────────────────────

    def safety_stock(self, service_level: float = 0.95) -> float:
        """
        Compute safety stock for a given cycle service level α.

            SS = z_α × σ_L   ... Silver et al. (1998), eq. 11.23

        where z_α = Φ^{-1}(α) is the standard normal quantile.

        Parameters
        ----------
        service_level : α ∈ (0, 1) — probability of no stockout per cycle

        Returns
        -------
        float — safety stock in units
        """
        if not 0 < service_level < 1:
            raise ValueError("Service level must be in (0, 1).")
        _, sig_L = self.lead_time_demand_moments()
        z_alpha  = stats.norm.ppf(service_level)   # Φ^{-1}(α)
        # SS = z_α × σ_L — normal approximation to safety stock
        SS = z_alpha * sig_L
        return max(0.0, round(SS, 2))

    # ── Reorder Point ─────────────────────────────────────────────────────────

    def reorder_point(self, service_level: float = 0.95) -> float:
        """
        Reorder point R = mean lead time demand + safety stock.

            R = μ_L + z_α × σ_L   ... Silver et al. (1998), eq. 11.24

        Returns
        -------
        float — reorder point in units
        """
        mu_L, _ = self.lead_time_demand_moments()
        SS       = self.safety_stock(service_level)
        R        = mu_L + SS
        return round(R, 2)

    # ── Order Quantity ────────────────────────────────────────────────────────

    def order_quantity(self) -> float:
        """
        Optimal order quantity Q via EOQ formula.

            Q = sqrt(2 × D_annual × K / h_annual)

        Returns
        -------
        float — order quantity in units
        """
        D_annual = self.mu_d * 365
        H_annual = self.h * 365
        # Q* = sqrt(2DS/H) — EOQ as order quantity for (Q,R) policy
        Q = np.sqrt(2 * D_annual * self.K / H_annual)
        return round(Q, 2)

    # ── Expected Annual Cost ──────────────────────────────────────────────────

    def expected_annual_cost(self, service_level: float = 0.95) -> Dict[str, float]:
        """
        Total expected annual cost of (Q,R) policy.

            TC = (D/Q)×K  +  (Q/2 + SS)×h×365  +  D×p×B(R)

        where B(R) = expected units short per cycle (lost sales).
        B(R) ≈ σ_L × G(z_α) — unit normal loss function.

            G(z) = φ(z) - z × [1 - Φ(z)]   ... Silver et al. eq. 11.33

        Returns
        -------
        dict: {ordering, holding, shortage, total}
        """
        Q  = self.order_quantity()
        SS = self.safety_stock(service_level)
        mu_L, sig_L = self.lead_time_demand_moments()
        D_annual    = self.mu_d * 365

        ordering_cost = (D_annual / Q) * self.K

        # Holding cost: cycle stock + safety stock
        holding_cost  = (Q / 2 + SS) * self.h * 365

        # Shortage cost: Expected backorders per cycle × (D/Q) cycles per year × p
        z = stats.norm.ppf(service_level)
        phi_z = stats.norm.pdf(z)
        Phi_z = stats.norm.cdf(z)
        # Unit normal loss function G(z) = φ(z) - z[1 - Φ(z)]
        G_z   = phi_z - z * (1 - Phi_z)
        # Expected units short per cycle
        EUS   = sig_L * G_z
        shortage_cost = (D_annual / Q) * self.p * EUS

        total = ordering_cost + holding_cost + shortage_cost
        return {
            "ordering":  round(ordering_cost, 2),
            "holding":   round(holding_cost, 2),
            "shortage":  round(shortage_cost, 2),
            "total":     round(total, 2),
            "Q":         Q,
            "R":         self.reorder_point(service_level),
            "SS":        SS,
        }

    # ── Cold Chain Extension: Excursion-Adjusted ROP ──────────────────────────

    def reorder_point_adjusted(
        self,
        service_level: float = 0.95,
        spoilage_prob: float = 0.05,
    ) -> Dict[str, float]:
        """
        Reorder point adjusted for cold chain temperature excursion spoilage.

        Extension logic:
            If P(excursion per shipment) = ρ, then a fraction ρ of each batch
            arrives with quality below threshold and is rejected.
            Effective order fulfillment: Q_eff = Q × (1 - ρ)

        To maintain the same service level, we must inflate R:
            R_adjusted = R_base / (1 - ρ)    ... (cold chain extension)

        This is equivalent to requiring the effective yield to cover μ_L + SS
        even after ρ fraction of batches are lost.

        Derivation:
            E[usable units received] = Q × (1 - ρ)
            To ensure P(D_L ≤ usable supply) ≥ α:
            R_adj = (μ_L + SS) / (1 - ρ)

        Parameters
        ----------
        service_level : α
        spoilage_prob : ρ — probability of a shipment being rejected due to excursion

        Returns
        -------
        dict: {R_base, R_adjusted, inflation_pct, spoilage_prob}
        """
        R_base  = self.reorder_point(service_level)
        # R_adj = R_base / (1 - ρ) — inflation to compensate for excursion losses
        if spoilage_prob >= 1.0:
            raise ValueError("spoilage_prob must be < 1.")
        R_adj   = R_base / (1.0 - spoilage_prob)

        return {
            "R_base":         round(R_base, 2),
            "R_adjusted":     round(R_adj, 2),
            "inflation_pct":  round(100 * (R_adj - R_base) / R_base, 2),
            "spoilage_prob":  spoilage_prob,
            "service_level":  service_level,
        }

    # ── Service Level Sweep ───────────────────────────────────────────────────

    def service_level_sweep(
        self,
        levels: Optional[List[float]] = None,
        spoilage_prob: float = 0.0,
    ) -> pd.DataFrame:
        """
        Sweep service level from lo to hi and compute SS, R, cost for each.

        Useful for understanding the cost of higher service levels —
        a key managerial insight in blood supply chain management.
        """
        if levels is None:
            levels = [0.80, 0.85, 0.90, 0.95, 0.98, 0.99]
        rows = []
        for alpha in levels:
            cost_dict = self.expected_annual_cost(alpha)
            R_adj = self.reorder_point_adjusted(alpha, spoilage_prob)["R_adjusted"]
            rows.append({
                "service_level":   alpha,
                "safety_stock":    cost_dict["SS"],
                "reorder_point":   cost_dict["R"],
                "R_adjusted":      R_adj,
                "annual_cost":     cost_dict["total"],
                "holding_cost":    cost_dict["holding"],
                "shortage_cost":   cost_dict["shortage"],
                "ordering_cost":   cost_dict["ordering"],
            })
        return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 4. FEFO POLICY (First Expired, First Out)
# ─────────────────────────────────────────────────────────────────────────────

class FEFOPolicy:
    """
    First Expired, First Out (FEFO) inventory issuing policy.

    WHO and blood bank guidelines mandate FEFO over FIFO because:
    - Blood products have hard expiry dates (5 days for platelets, 42 for RBCs)
    - Issuing the nearest-to-expiry unit first minimises waste
    - FIFO can lead to systematic expiry of early batches during low-demand periods

    Reference: WHO (2019). "Manual on the Management, Maintenance and Use of
               Blood Cold Chain Equipment." WHO/EMP/2019.02

    This class manages a batch inventory as a list of (quantity, expiry_date) tuples
    and implements FEFO issuing with expiry tracking.

    FEFO vs FIFO comparison:
        FEFO always issues the earliest-expiry batch first.
        FIFO issues the earliest-received batch first (may not be earliest-expiry
        if batches have variable ages at receipt).
    """

    def __init__(self, product_name: str, shelf_life_days: int, seed: int = DEFAULT_SEED):
        """
        Parameters
        ----------
        product_name   : e.g. 'Platelet', 'PackedRBC', 'WholeBlood'
        shelf_life_days: maximum shelf life in days
        seed           : random seed
        """
        self.product_name   = product_name
        self.shelf_life_days = shelf_life_days
        self.rng            = np.random.default_rng(seed)
        # Batch format: list of dicts {quantity, manufacture_date, expiry_date, batch_id}
        self._batches: List[Dict] = []
        self._batch_counter = 0
        # History tracking
        self.history: List[Dict] = []

    def add_batch(self, quantity: int, current_day: int, manufacture_day: Optional[int] = None) -> str:
        """
        Add a new batch to inventory.

        Parameters
        ----------
        quantity      : number of units in batch
        current_day   : current simulation day (day of receipt)
        manufacture_day: day manufactured (if None, assumes manufactured today)

        Returns batch_id.
        """
        if manufacture_day is None:
            manufacture_day = current_day
        age_at_receipt  = current_day - manufacture_day
        remaining_life  = self.shelf_life_days - age_at_receipt
        if remaining_life <= 0:
            warnings.warn(f"Batch received with 0 remaining shelf life (age={age_at_receipt}d).")
            return None
        expiry_day = current_day + remaining_life
        batch_id   = f"{self.product_name}_B{self._batch_counter:04d}"
        self._batch_counter += 1
        self._batches.append({
            "batch_id":        batch_id,
            "quantity":        quantity,
            "manufacture_day": manufacture_day,
            "receipt_day":     current_day,
            "expiry_day":      expiry_day,
            "remaining_life":  remaining_life,
        })
        return batch_id

    def _sort_batches_fefo(self):
        """Sort batches by expiry_day ascending (FEFO order)."""
        self._batches.sort(key=lambda b: b["expiry_day"])

    def _sort_batches_fifo(self):
        """Sort batches by receipt_day ascending (FIFO order)."""
        self._batches.sort(key=lambda b: b["receipt_day"])

    def issue_units(self, n_units: int, current_day: int, policy: str = "FEFO") -> Dict:
        """
        Issue n_units using FEFO or FIFO policy.

        FEFO: Comment: "FEFO is preferred in pharmaceutical/blood supply chains
        (WHO guidelines) over FIFO to minimise expiry waste. Platelets (5-day
        shelf life) especially benefit — a 1-day improvement in average age at
        issue can reduce wastage by 15-20%."

        Parameters
        ----------
        n_units    : units requested
        current_day: current simulation day
        policy     : 'FEFO' or 'FIFO'

        Returns
        -------
        dict: {issued, units_short, batches_used, avg_remaining_life}
        """
        if policy == "FEFO":
            self._sort_batches_fefo()
        elif policy == "FIFO":
            self._sort_batches_fifo()
        else:
            raise ValueError("policy must be 'FEFO' or 'FIFO'")

        units_needed  = n_units
        units_issued  = 0
        batches_used  = []
        total_rem_life = 0.0

        for batch in self._batches:
            if units_needed <= 0:
                break
            if batch["expiry_day"] <= current_day:
                continue   # expired (will be cleaned by waste_units)
            can_issue = min(batch["quantity"], units_needed)
            batch["quantity"] -= can_issue
            units_issued  += can_issue
            units_needed  -= can_issue
            total_rem_life += can_issue * (batch["expiry_day"] - current_day)
            batches_used.append((batch["batch_id"], can_issue))

        # Remove depleted batches
        self._batches = [b for b in self._batches if b["quantity"] > 0]

        avg_rem_life = (total_rem_life / units_issued) if units_issued > 0 else 0.0
        result = {
            "issued":             units_issued,
            "units_short":        max(0, n_units - units_issued),
            "batches_used":       batches_used,
            "avg_remaining_life": round(avg_rem_life, 2),
            "policy":             policy,
        }
        self.history.append({"day": current_day, "event": "issue", **result})
        return result

    def waste_units(self, current_day: int) -> Dict:
        """
        Remove expired units and return wastage statistics.

        All units with expiry_day <= current_day are considered expired.
        Called daily in the simulation expiry_check process.

        Returns
        -------
        dict: {units_wasted, batches_expired, wastage_cost_per_unit_info}
        """
        expired_batches = [b for b in self._batches if b["expiry_day"] <= current_day]
        total_wasted    = sum(b["quantity"] for b in expired_batches)
        self._batches   = [b for b in self._batches if b["expiry_day"] > current_day]

        result = {
            "units_wasted":    total_wasted,
            "batches_expired": len(expired_batches),
            "expired_batch_ids": [b["batch_id"] for b in expired_batches],
            "day":             current_day,
        }
        if total_wasted > 0:
            self.history.append({"event": "waste", **result})
        return result

    def current_inventory(self, current_day: int) -> Dict:
        """Return current inventory summary after removing expired units."""
        self.waste_units(current_day)
        self._sort_batches_fefo()
        total_units = sum(b["quantity"] for b in self._batches)
        avg_rem     = (
            sum(b["quantity"] * (b["expiry_day"] - current_day) for b in self._batches)
            / total_units if total_units > 0 else 0.0
        )
        return {
            "total_units":        total_units,
            "n_batches":          len(self._batches),
            "avg_remaining_life": round(avg_rem, 2),
            "batches":            list(self._batches),
        }

    @staticmethod
    def compare_fefo_vs_fifo(
        n_days: int = 30,
        mean_demand: float = 20,
        std_demand: float = 5,
        order_quantity: int = 25,
        order_interval_days: int = 5,
        shelf_life_days: int = 5,
        seed: int = DEFAULT_SEED,
    ) -> Dict:
        """
        Simulate FEFO and FIFO over n_days and compare wastage.

        Comment: "FEFO is preferred in pharmaceutical/blood supply chains
        (WHO guidelines) over FIFO to minimise expiry waste. Platelets
        (5-day shelf life) especially benefit from FEFO: issuing nearest-
        to-expiry first prevents older stock from silently expiring behind
        newer batches."

        Returns
        -------
        dict: {fefo_waste, fifo_waste, waste_reduction_pct, daily_comparison_df}
        """
        rng = np.random.default_rng(seed)
        results = {}

        for policy in ["FEFO", "FIFO"]:
            inv   = FEFOPolicy("TestProduct", shelf_life_days, seed=seed)
            waste_total  = 0
            shortage_total = 0
            daily_waste  = []

            for day in range(n_days):
                # Replenishment every order_interval_days
                if day % order_interval_days == 0:
                    inv.add_batch(order_quantity, day)

                # Demand
                demand = max(0, int(rng.normal(mean_demand, std_demand)))
                issue  = inv.issue_units(demand, day, policy=policy)
                shortage_total += issue["units_short"]

                # Expiry check
                w = inv.waste_units(day)
                waste_total += w["units_wasted"]
                daily_waste.append({"day": day, "waste": w["units_wasted"],
                                    "shortage": issue["units_short"], "policy": policy})

            results[policy] = {
                "total_waste":    waste_total,
                "total_shortage": shortage_total,
                "daily":          daily_waste,
            }

        fefo_w  = results["FEFO"]["total_waste"]
        fifo_w  = results["FIFO"]["total_waste"]
        reduction = 100 * (fifo_w - fefo_w) / max(fifo_w, 1)

        df_fefo = pd.DataFrame(results["FEFO"]["daily"])
        df_fifo = pd.DataFrame(results["FIFO"]["daily"])
        df_cmp  = df_fefo[["day", "waste"]].merge(
            df_fifo[["day", "waste"]], on="day", suffixes=("_FEFO", "_FIFO")
        )

        return {
            "fefo_waste":           fefo_w,
            "fifo_waste":           fifo_w,
            "waste_reduction_pct":  round(reduction, 2),
            "fefo_shortage":        results["FEFO"]["total_shortage"],
            "fifo_shortage":        results["FIFO"]["total_shortage"],
            "daily_comparison_df":  df_cmp,
        }
