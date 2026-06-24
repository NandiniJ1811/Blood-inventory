"""
models/perishability.py
=======================
Perishable product model integrating inventory theory with TTI quality degradation.

Key insight: perishability creates a coupled trade-off:
    - Larger orders → lower stockout risk (Newsvendor logic: lower underage cost)
    - Larger orders → higher wastage (units expire before demand arrives)
    - Temperature excursions further reduce effective shelf life

Optimal order quantity must jointly balance:
    E[holding] + E[shortage] + E[wastage] → minimised over Q

This extends classical Newsvendor (Nahmias 1975) to include an explicit
wastage cost term — a critical addition for blood products (particularly
platelets with 5-day shelf life) and vaccines.

Reference:
    Nahmias, S. (1975). "Optimal ordering policies for perishable inventory."
    Operations Research, 23(4), 735-749.
    Nahmias, S. (2011). "Perishable Inventory Systems." Springer.

Author: IIT Kharagpur, Dept. of Industrial and Systems Engineering
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from scipy import stats, optimize

from .temperature import TimeTemperatureIndex

DEFAULT_SEED = 42

# ─────────────────────────────────────────────────────────────────────────────
# Product catalogue — blood products (WHO specifications)
# ─────────────────────────────────────────────────────────────────────────────

BLOOD_PRODUCTS = {
    "WholeBlood": {
        "shelf_life_days": 35,
        "storage_temp_c": 4.0,
        "temp_tolerance_c": 2.0,   # acceptable deviation
        "unit_cost": 1200,          # ₹ per unit
        "activation_energy": 68_000,
        "description": "Whole blood — 35-day shelf life at 2-6°C",
    },
    "PackedRBC": {
        "shelf_life_days": 42,
        "storage_temp_c": 4.0,
        "temp_tolerance_c": 2.0,
        "unit_cost": 1500,
        "activation_energy": 70_000,
        "description": "Packed red blood cells — 42-day shelf life at 2-6°C",
    },
    "Platelet": {
        "shelf_life_days": 5,
        "storage_temp_c": 22.0,    # platelets stored at 20-24°C on agitator
        "temp_tolerance_c": 2.0,
        "unit_cost": 2500,
        "activation_energy": 55_000,
        "description": "Platelets — 5-day shelf life at 20-24°C; most critical",
    },
}


@dataclass
class ProductBatch:
    """A single received batch of perishable product."""
    batch_id:          str
    product_type:      str
    quantity:          int
    manufacture_day:   float   # simulation day
    receipt_day:       float   # simulation day when received
    base_shelf_life_h: float   # nominal shelf life in hours (at reference temp)
    current_quality:   float = 1.0
    temp_history:      List[float] = field(default_factory=list)
    effective_expiry_day: Optional[float] = None  # set after TTI assessment

    def age_days(self, current_day: float) -> float:
        return current_day - self.receipt_day

    def is_expired(self, current_day: float) -> bool:
        if self.effective_expiry_day is not None:
            return current_day >= self.effective_expiry_day
        return (current_day - self.manufacture_day) * 24 >= self.base_shelf_life_h


class PerishableProduct:
    """
    Inventory model for perishable cold chain products integrating
    TTI-based shelf life reduction and extended Newsvendor optimisation.

    Manages:
        - A batch inventory list: [(quantity, manufacture_date, current_quality)]
        - Effective shelf life computation via Arrhenius/TTI model
        - Wastage rate estimation for given (Q, demand) combination
        - Extended Newsvendor: min E[holding + shortage + wastage]

    Products supported: WholeBlood (35d), PackedRBC (42d), Platelet (5d)

    Comment: "Perishability creates a trade-off: larger orders reduce
    stockout risk (Newsvendor logic) but increase wastage — optimal Q*
    balances both. For platelets (5-day shelf life), this trade-off is
    extreme: over-ordering by even 2 days' demand causes 40% wastage."
    """

    def __init__(
        self,
        product_type: str = "PackedRBC",
        seed: int = DEFAULT_SEED,
    ):
        """
        Parameters
        ----------
        product_type : 'WholeBlood', 'PackedRBC', or 'Platelet'
        seed         : random seed
        """
        if product_type not in BLOOD_PRODUCTS:
            raise ValueError(
                f"Unknown product type '{product_type}'. "
                f"Choose from: {list(BLOOD_PRODUCTS.keys())}"
            )
        self.product_type = product_type
        self.spec         = BLOOD_PRODUCTS[product_type]
        self.shelf_life_days = self.spec["shelf_life_days"]
        self.shelf_life_h    = self.shelf_life_days * 24.0
        self.rng          = np.random.default_rng(seed)
        self._batches: List[ProductBatch] = []
        self._batch_counter = 0

        # TTI model calibrated for this product
        # Reference rate = 1/shelf_life_h (1 unit of quality consumed over full shelf life at ref temp)
        self.tti = TimeTemperatureIndex(
            activation_energy=self.spec["activation_energy"],
            reference_temp_c=self.spec["storage_temp_c"],
            reference_rate=1.0 / self.shelf_life_h,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Batch management
    # ─────────────────────────────────────────────────────────────────────────

    def receive_batch(
        self,
        quantity: int,
        current_day: float,
        manufacture_day: Optional[float] = None,
        temp_history: Optional[List[float]] = None,
    ) -> str:
        """
        Add a new batch to inventory and assess its effective shelf life via TTI.

        Parameters
        ----------
        quantity        : units in batch
        current_day     : simulation day of receipt
        manufacture_day : day manufactured (None = today)
        temp_history    : list of temperatures (°C) during transit (minute-by-minute)

        Returns
        -------
        batch_id or None if quality below threshold
        """
        if manufacture_day is None:
            manufacture_day = current_day
        batch_id = f"{self.product_type}_B{self._batch_counter:04d}"
        self._batch_counter += 1

        batch = ProductBatch(
            batch_id=batch_id,
            product_type=self.product_type,
            quantity=quantity,
            manufacture_day=manufacture_day,
            receipt_day=current_day,
            base_shelf_life_h=self.shelf_life_h,
            current_quality=1.0,
            temp_history=temp_history or [],
        )

        # Assess effective shelf life using TTI if temperature history provided
        if temp_history:
            quality_series = self.tti.quality_decay(np.array(temp_history), dt_minutes=1.0)
            batch.current_quality = float(quality_series[-1])

            remaining_h = self.tti.residual_shelf_life(
                np.array(temp_history),
                self.shelf_life_h,
                dt_minutes=1.0,
            )
            batch.effective_expiry_day = current_day + remaining_h / 24.0
        else:
            batch.current_quality   = 1.0
            batch.effective_expiry_day = manufacture_day + self.shelf_life_days

        self._batches.append(batch)
        return batch_id

    def effective_shelf_life(self, temp_history: List[float]) -> float:
        """
        Compute effective remaining shelf life in hours after temperature exposure.

        Uses TTI Arrhenius model to compute cumulative degradation from
        temp_history, then estimates remaining time at reference temperature.

        Parameters
        ----------
        temp_history : list of temperature readings (°C), 1-minute intervals

        Returns
        -------
        float — effective remaining shelf life in hours
        """
        return self.tti.residual_shelf_life(
            np.array(temp_history),
            self.shelf_life_h,
            dt_minutes=1.0,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Wastage analysis
    # ─────────────────────────────────────────────────────────────────────────

    def wastage_rate(
        self,
        order_quantity: float,
        demand_mean: float,
        demand_std: float,
        n_simulations: int = 5000,
    ) -> Dict:
        """
        Estimate expected fraction wasted per replenishment cycle via Monte Carlo.

        For a perishable product with shelf life m:
            - Order Q units
            - Demand D ~ N(μ, σ²)
            - Units wasted = max(Q - D, 0) (conservative single-period approximation)
            - Wastage rate = E[max(Q-D,0)] / Q

        Comment: "Perishability creates a trade-off: larger orders reduce
        stockout risk (Newsvendor logic) but increase wastage — optimal Q*
        balances both. This is the fundamental tension in blood inventory
        management."

        Parameters
        ----------
        order_quantity : Q — units ordered per cycle
        demand_mean    : μ
        demand_std     : σ
        n_simulations  : Monte Carlo replications

        Returns
        -------
        dict: {wastage_rate, expected_wasted, expected_shortage, service_level}
        """
        demands   = self.rng.normal(demand_mean, demand_std, n_simulations)
        demands   = np.maximum(0, demands)

        # Units wasted = max(Q - D, 0) per period
        wasted    = np.maximum(order_quantity - demands, 0)
        # Units short = max(D - Q, 0)
        short     = np.maximum(demands - order_quantity, 0)

        wastage_rate    = float(np.mean(wasted)) / order_quantity if order_quantity > 0 else 0
        service_level   = float(np.mean(demands <= order_quantity))

        return {
            "order_quantity":    order_quantity,
            "wastage_rate":      round(wastage_rate, 4),
            "expected_wasted":   round(float(np.mean(wasted)), 2),
            "expected_shortage": round(float(np.mean(short)), 2),
            "service_level":     round(service_level, 4),
            "demand_mean":       demand_mean,
            "demand_std":        demand_std,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Extended Newsvendor with wastage cost
    # ─────────────────────────────────────────────────────────────────────────

    def optimal_order_under_perishability(
        self,
        demand_mean: float,
        demand_std: float,
        holding_cost: float,
        shortage_cost: float,
        wastage_cost: float,
        shelf_life_days: Optional[int] = None,
    ) -> Dict:
        """
        Extended Newsvendor minimising: E[holding + shortage + wastage].

        Objective:
            C(Q) = h × E[max(Q-D,0)] + p × E[max(D-Q,0)] + w × E[max(Q-D,0)]
                 = (h + w) × E[max(Q-D,0)] + p × E[max(D-Q,0)]

        where:
            h = holding cost per unit per period
            p = shortage penalty per unit short
            w = wastage cost per unit wasted (= unit_cost for blood products)

        This is structurally identical to the Newsvendor with modified Co:
            Co_eff = h + w   (combined overage cost)
            Cu     = p
            Q*_perishable = Φ^{-1}(p / (p + h + w)) × σ + μ

        For blood products, w >> h (blood is expensive to produce),
        so wastage cost dominates the overage penalty and drives Q* down
        significantly compared to a non-perishable product.

        Solve numerically via scipy.optimize.minimize_scalar for
        generalisation (handles non-normal demand distributions).

        Parameters
        ----------
        demand_mean   : μ
        demand_std    : σ
        holding_cost  : h — cost per unit held per period
        shortage_cost : p — penalty per unit short
        wastage_cost  : w — cost per unit wasted (typically ≥ unit_cost)
        shelf_life_days: override (else uses product's shelf life)

        Returns
        -------
        dict: {Q_star, cost_at_Q_star, service_level_at_Q_star,
               wastage_rate_at_Q_star, breakdown comparison}
        """
        Co_eff = holding_cost + wastage_cost   # combined overage cost

        def total_cost(Q: float) -> float:
            """Expected total cost at order quantity Q."""
            if Q <= 0:
                return 1e12
            # E[max(Q-D,0)] and E[max(D-Q,0)] under Normal demand
            z        = (Q - demand_mean) / (demand_std + 1e-9)
            phi_z    = stats.norm.pdf(z)
            Phi_z    = stats.norm.cdf(z)
            E_over   = (Q - demand_mean) * Phi_z + demand_std * phi_z
            E_under  = demand_std * (phi_z - z * (1 - Phi_z))
            return Co_eff * E_over + shortage_cost * E_under

        # Solve for Q* ∈ [0, demand_mean + 4*demand_std]
        Q_lo = max(0.01, demand_mean - 3 * demand_std)
        Q_hi = demand_mean + 4 * demand_std

        result = optimize.minimize_scalar(total_cost, bounds=(Q_lo, Q_hi), method="bounded")
        Q_star = max(0.0, result.x)

        # Analytical check: Q* = Φ^{-1}(p/(p+Co_eff)) × σ + μ
        cr_extended = shortage_cost / (shortage_cost + Co_eff)
        Q_analytical = stats.norm.ppf(cr_extended, loc=demand_mean, scale=demand_std)

        # Compute breakdown at Q*
        wastage_info = self.wastage_rate(Q_star, demand_mean, demand_std, n_simulations=2000)

        # Compare: classic Newsvendor (no wastage) vs extended
        cr_classic    = shortage_cost / (shortage_cost + holding_cost)
        Q_classic     = stats.norm.ppf(cr_classic, loc=demand_mean, scale=demand_std)

        return {
            "Q_star_extended":     round(Q_star, 2),
            "Q_star_classic":      round(max(0, Q_classic), 2),
            "Q_analytical":        round(max(0, Q_analytical), 2),
            "cost_at_Q_star":      round(result.fun, 2),
            "cr_extended":         round(cr_extended, 4),
            "cr_classic":          round(cr_classic, 4),
            "Co_effective":        round(Co_eff, 2),
            "service_level":       round(wastage_info["service_level"], 4),
            "wastage_rate":        round(wastage_info["wastage_rate"], 4),
            "expected_wasted":     round(wastage_info["expected_wasted"], 2),
            "Q_reduction_pct":     round(100 * (Q_classic - Q_star) / max(Q_classic, 1), 2),
            "product_type":        self.product_type,
            "shelf_life_days":     shelf_life_days or self.shelf_life_days,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Trade-off analysis
    # ─────────────────────────────────────────────────────────────────────────

    def order_quantity_tradeoff_plot(
        self,
        demand_mean: float,
        demand_std: float,
        holding_cost: float = 3.0,
        shortage_cost: float = 50.0,
        wastage_cost: float = 15.0,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """
        Plot the classic perishability trade-off curve:
        Order quantity Q (x-axis) vs. wastage rate and service level (y-axes).

        Shows clearly why simply maximising service level by ordering more
        leads to unacceptable wastage in perishable product supply chains.
        """
        Q_range = np.linspace(
            max(1, demand_mean * 0.3),
            demand_mean * 2.5,
            60,
        )
        waste_rates    = []
        service_levels = []
        total_costs    = []

        for Q in Q_range:
            wr = self.wastage_rate(Q, demand_mean, demand_std, n_simulations=500)
            waste_rates.append(wr["wastage_rate"])
            service_levels.append(wr["service_level"])

            Co_eff = holding_cost + wastage_cost
            z      = (Q - demand_mean) / (demand_std + 1e-9)
            phi_z  = stats.norm.pdf(z)
            Phi_z  = stats.norm.cdf(z)
            E_over = (Q - demand_mean) * Phi_z + demand_std * phi_z
            E_und  = demand_std * (phi_z - z * (1 - Phi_z))
            total_costs.append(Co_eff * E_over + shortage_cost * E_und)

        # Optimal Q
        opt_res = self.optimal_order_under_perishability(
            demand_mean, demand_std, holding_cost, shortage_cost, wastage_cost
        )
        Q_star = opt_res["Q_star_extended"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: Service level vs Wastage rate (trade-off frontier)
        axes[0].plot(
            [wr * 100 for wr in waste_rates],
            [sl * 100 for sl in service_levels],
            color="#2c7bb6", lw=2.5,
        )
        # Highlight Q* on the frontier
        idx_star = np.argmin(np.abs(np.array(Q_range) - Q_star))
        axes[0].scatter(
            [waste_rates[idx_star] * 100], [service_levels[idx_star] * 100],
            color="red", s=100, zorder=5, label=f"Q* = {Q_star:.0f}"
        )
        axes[0].set_xlabel("Wastage Rate (%)")
        axes[0].set_ylabel("Service Level (%)")
        axes[0].set_title(
            f"Perishability Trade-off — {self.product_type}\n"
            "(Shelf life = {self.shelf_life_days}d)"
        )
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Right: Total cost curve
        axes[1].plot(Q_range, total_costs, color="#d7191c", lw=2)
        axes[1].axvline(Q_star, color="black", lw=1.5, linestyle="--",
                         label=f"Q* = {Q_star:.0f}")
        axes[1].set_xlabel("Order Quantity Q")
        axes[1].set_ylabel("Expected Total Cost")
        axes[1].set_title("Extended Newsvendor Cost Curve\nE[holding + shortage + wastage]")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.suptitle(
            f"{self.product_type} — Perishability-Aware Inventory Optimisation\n"
            f"μ={demand_mean}, σ={demand_std}, h={holding_cost}, p={shortage_cost}, w={wastage_cost}",
            fontsize=9,
        )
        plt.tight_layout()
        return fig

    # ─────────────────────────────────────────────────────────────────────────
    # Current inventory summary
    # ─────────────────────────────────────────────────────────────────────────

    def current_inventory_summary(self, current_day: float) -> Dict:
        """Return summary of current batches in inventory."""
        valid = [b for b in self._batches if not b.is_expired(current_day)]
        expired = [b for b in self._batches if b.is_expired(current_day)]

        total_units = sum(b.quantity for b in valid)
        avg_quality = (
            np.mean([b.current_quality for b in valid]) if valid else 0.0
        )
        avg_rem_life = (
            np.mean([(b.effective_expiry_day or b.manufacture_day + b.shelf_life_days) - current_day
                     for b in valid]) if valid else 0.0
        )

        return {
            "product_type":     self.product_type,
            "total_units":      total_units,
            "n_batches":        len(valid),
            "n_expired_batches": len(expired),
            "units_to_waste":   sum(b.quantity for b in expired),
            "avg_quality":      round(float(avg_quality), 4),
            "avg_remaining_life_days": round(float(avg_rem_life), 2),
            "current_day":      current_day,
        }
