"""
simulation/metrics.py
=====================
KPI collection, aggregation, and reporting for the cold chain simulation.

Tracks per-node, per-product, per-week statistics across simulation runs.
Supports multi-replication averaging with confidence intervals

KPIs tracked:
  - Service level (fill rate) per node per week
  - Inventory level time series
  - Temperature excursions (count, severity, duration)
  - Units wasted (expiry + quality rejection)
  - Cost breakdown: holding, ordering, shortage, wastage, emergency, maintenance
  - Average residual shelf life at point of issue
  - Cold chain compliance rate (% transits with zero critical excursion)

Author: IIT Kharagpur, Dept. of Industrial and Systems Engineering
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from dataclasses import dataclass, field

from .events import SimEvent, EventType


# ─────────────────────────────────────────────────────────────────────────────
# Cost parameters (used for cost aggregation)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_COSTS = {
    "holding_per_unit_day":          3.50,    # ₹/unit/day
    "ordering_fixed":                500.0,   # ₹/order
    "shortage_per_unit":             200.0,   # ₹/unit short
    "wastage_per_unit":              1500.0,  # ₹/unit wasted (unit cost)
    "emergency_resupply_fixed":      5000.0,  # ₹/emergency call
    "emergency_resupply_per_unit":   200.0,   # ₹/unit premium
    "equipment_repair_per_hour":     800.0,   # ₹/hour downtime
}


@dataclass
class WeeklyKPI:
    """KPI snapshot for one week at one node."""
    week:             int
    node_id:          int
    product:          str
    demand_total:     int   = 0
    fulfilled_total:  int   = 0
    shortage_total:   int   = 0
    orders_placed:    int   = 0
    orders_rejected:  int   = 0
    units_wasted:     int   = 0
    n_excursions_warn: int  = 0
    n_excursions_crit: int  = 0
    n_equipment_failures: int = 0
    n_emergency_resupply: int = 0
    avg_inventory:    float = 0.0
    avg_rem_shelf_life: float = 0.0

    @property
    def service_level(self) -> float:
        return self.fulfilled_total / max(self.demand_total, 1)

    @property
    def wastage_rate(self) -> float:
        total_received = self.fulfilled_total + self.units_wasted
        return self.units_wasted / max(total_received, 1)


class MetricsCollector:
    """
    Collects, stores, and aggregates simulation events into KPIs.

    Design:
        - All raw events are appended to self.event_log (list of dicts)
        - Inventory levels are tracked as a time series per node
        - Cost totals are updated incrementally
        - Weekly KPIs are computed by slicing the event log

    Usage:
        metrics = MetricsCollector(warmup_days=60)
        metrics.log_event(event)
        metrics.log_inventory(time=10.5, node_id=3, units=45, product='PackedRBC')
        summary = metrics.summary()
    """

    def __init__(
        self,
        warmup_days: float = 60.0,
        cost_params: Optional[Dict] = None,
        products: Optional[List[str]] = None,
        nodes: Optional[List[int]] = None,
    ):
        """
        Parameters
        ----------
        warmup_days  : first N days excluded from statistics (transient phase)
        cost_params  : cost parameters dict (defaults to DEFAULT_COSTS)
        products     : list of product types being simulated
        nodes        : list of node IDs in the network
        """
        self.warmup_hours = warmup_days * 24.0
        self.costs        = {**DEFAULT_COSTS, **(cost_params or {})}
        self.products     = products or ["PackedRBC"]
        self.nodes        = nodes or [0, 1, 2, 3]

        # Raw event log (post-warmup only — populated by log_event)
        self.event_log:    List[Dict] = []
        # Full event log (including warmup — for debugging)
        self._all_events:  List[Dict] = []

        # Inventory time series: {(node_id, product): [(time, units)]}
        self.inv_series: Dict[Tuple, List[Tuple[float, int]]] = defaultdict(list)

        # Temperature series: {node_id: [(time, temp)]}
        self.temp_series: Dict[int, List[Tuple[float, float]]] = defaultdict(list)

        # Running cost totals (post-warmup)
        self._costs: Dict[str, float] = defaultdict(float)

        # Quality at issue: list of (time, quality_index) for post-warmup fulfilled demand
        self.quality_at_issue: List[Tuple[float, float]] = []

        # Residual shelf life at issue: list of (time, days)
        self.rem_shelf_life_at_issue: List[Tuple[float, float]] = []

        # Compliance tracking: each transit → (compliant: bool)
        self.transit_log: List[Dict] = []

    # ─────────────────────────────────────────────────────────────────────────
    # Logging methods
    # ─────────────────────────────────────────────────────────────────────────

    def log_event(self, event: SimEvent):
        """Log a simulation event. Filters out warmup period for main stats."""
        d = event.to_dict()
        self._all_events.append(d)
        if event.time >= self.warmup_hours:
            self.event_log.append(d)
            self._update_cost_from_event(d)

    def log_inventory(self, time: float, node_id: int, units: int, product: str):
        """Log a point-in-time inventory level reading."""
        self.inv_series[(node_id, product)].append((time, units))

    def log_temperature(self, time: float, node_id: int, temp_c: float):
        """Log a temperature reading at a node."""
        self.temp_series[node_id].append((time, temp_c))

    def log_quality_at_issue(self, time: float, quality: float, rem_life_days: float):
        """Log the quality index and residual shelf life of units issued."""
        if time >= self.warmup_hours:
            self.quality_at_issue.append((time, quality))
            self.rem_shelf_life_at_issue.append((time, rem_life_days))

    def log_transit(self, time: float, from_node: int, to_node: int,
                    product: str, had_critical_excursion: bool):
        """Log outcome of a transit leg for compliance rate calculation."""
        self.transit_log.append({
            "time":                  time,
            "from_node":             from_node,
            "to_node":               to_node,
            "product":               product,
            "had_critical_excursion": had_critical_excursion,
        })

    def _update_cost_from_event(self, event_dict: Dict):
        """Incrementally update cost totals from an event."""
        etype = event_dict.get("event_type", "")
        c     = self.costs

        if etype == EventType.STOCKOUT:
            self._costs["shortage"] += (
                event_dict.get("shortage", 0) * c["shortage_per_unit"]
            )
        elif etype == EventType.DEMAND_PARTIAL:
            self._costs["shortage"] += (
                event_dict.get("shortage", 0) * c["shortage_per_unit"]
            )
        elif etype == EventType.ORDER_PLACED:
            self._costs["ordering"] += c["ordering_fixed"]
        elif etype == EventType.EXPIRY_WASTE:
            self._costs["wastage"] += (
                event_dict.get("units_wasted", 0) * c["wastage_per_unit"]
            )
        elif etype == EventType.ORDER_REJECTED:
            self._costs["wastage"] += (
                event_dict.get("order_quantity", 0) * c["wastage_per_unit"]
            )
        elif etype == EventType.EMERGENCY_RESUPPLY:
            self._costs["emergency"] += (
                c["emergency_resupply_fixed"]
                + event_dict.get("units_ordered", 0) * c["emergency_resupply_per_unit"]
            )
        elif etype == EventType.EQUIPMENT_FAILURE:
            self._costs["maintenance"] += (
                event_dict.get("repair_duration_h", 4.0) * c["equipment_repair_per_hour"]
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Holding cost (computed from inventory time series)
    # ─────────────────────────────────────────────────────────────────────────

    def compute_holding_cost(self) -> float:
        """
        Compute total holding cost from inventory time series.

        Uses trapezoidal integration: ∫ I(t) dt × holding_cost_per_unit_day

        Returns
        -------
        float — total holding cost (₹)
        """
        total_holding = 0.0
        h_per_unit_day = self.costs["holding_per_unit_day"]

        for (node_id, product), series in self.inv_series.items():
            if not series:
                continue
            times = np.array([s[0] for s in series])
            units = np.array([s[1] for s in series], dtype=float)

            # Filter to post-warmup only
            mask  = times >= self.warmup_hours
            if mask.sum() < 2:
                continue
            t_filt = times[mask]
            u_filt = units[mask]

            # Trapezoidal area in unit-hours → convert to unit-days → × cost
            area_unit_hours = np.trapz(u_filt, t_filt)
            area_unit_days  = area_unit_hours / 24.0
            total_holding  += area_unit_days * h_per_unit_day

        self._costs["holding"] = total_holding
        return total_holding

    # ─────────────────────────────────────────────────────────────────────────
    # KPI computation
    # ─────────────────────────────────────────────────────────────────────────

    def _events_df(self) -> pd.DataFrame:
        """Return post-warmup events as a DataFrame."""
        if not self.event_log:
            return pd.DataFrame()
        return pd.DataFrame(self.event_log)

    def service_level(self, node_id: Optional[int] = None) -> float:
        """
        Fill-rate service level = fulfilled / (fulfilled + shortage).

        Parameters
        ----------
        node_id : if None, aggregates across all nodes

        Returns
        -------
        float ∈ [0, 1]
        """
        df = self._events_df()
        if df.empty:
            return 0.0
        if node_id is not None:
            df = df[df["node_id"] == node_id]

        demand_events = df[df["event_type"].isin([
            EventType.DEMAND_FULFILLED,
            EventType.DEMAND_PARTIAL,
            EventType.STOCKOUT,
        ])]
        total_demand    = demand_events.get("demand", pd.Series(dtype=float)).sum()
        total_fulfilled = demand_events.get("fulfilled", pd.Series(dtype=float)).sum()
        return float(total_fulfilled / max(total_demand, 1))

    def total_cost_breakdown(self) -> Dict[str, float]:
        """
        Return total cost breakdown across all cost categories.

        Holding cost is computed from inventory time series (trapezoidal integration).
        All other costs are accumulated from events.

        Returns
        -------
        dict: {holding, ordering, shortage, wastage, emergency, maintenance, total}
        """
        self.compute_holding_cost()
        breakdown = {k: round(v, 2) for k, v in self._costs.items()}
        breakdown["total"] = round(sum(breakdown.values()), 2)
        return breakdown

    def weekly_service_levels(
        self,
        node_id: int = 3,
        product: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Compute service level for each week of post-warmup simulation.

        Parameters
        ----------
        node_id : node to analyse (default 3 = hospital)
        product : filter by product type (None = all products)

        Returns
        -------
        pd.DataFrame: {week, demand, fulfilled, shortage, service_level}
        """
        df = self._events_df()
        if df.empty:
            return pd.DataFrame()

        df = df[df["node_id"] == node_id]
        if product:
            df = df[df["product"] == product]

        demand_df = df[df["event_type"].isin([
            EventType.DEMAND_FULFILLED,
            EventType.DEMAND_PARTIAL,
            EventType.STOCKOUT,
        ])].copy()

        if demand_df.empty:
            return pd.DataFrame()

        # Convert time (hours) to week index
        demand_df["week"] = ((demand_df["time"] - self.warmup_hours) / (7 * 24)).astype(int)

        weekly = demand_df.groupby("week").agg(
            demand    = ("demand",    "sum"),
            fulfilled = ("fulfilled", "sum"),
            shortage  = ("shortage",  "sum"),
        ).reset_index()
        weekly["service_level"] = weekly["fulfilled"] / weekly["demand"].clip(lower=1)
        return weekly

    def excursion_summary(self) -> Dict:
        """
        Summarise temperature excursion events.

        Returns
        -------
        dict: {n_warning, n_critical, total_duration_min, mean_duration_min,
               most_affected_node}
        """
        df = self._events_df()
        if df.empty:
            return {"n_warning": 0, "n_critical": 0}

        exc_df = df[df["event_type"].isin([
            EventType.TEMP_EXCURSION_WARN,
            EventType.TEMP_EXCURSION_CRIT,
        ])]

        n_warn  = int((exc_df["event_type"] == EventType.TEMP_EXCURSION_WARN).sum())
        n_crit  = int((exc_df["event_type"] == EventType.TEMP_EXCURSION_CRIT).sum())

        durations = exc_df.get("duration_minutes", pd.Series(dtype=float))
        total_dur = float(durations.sum()) if not durations.empty else 0.0
        mean_dur  = float(durations.mean()) if not durations.empty else 0.0

        most_affected = None
        if not exc_df.empty and "node_id" in exc_df.columns:
            most_affected = int(exc_df["node_id"].value_counts().idxmax())

        return {
            "n_warning":            n_warn,
            "n_critical":           n_crit,
            "total_duration_min":   round(total_dur, 1),
            "mean_duration_min":    round(mean_dur, 1),
            "most_affected_node":   most_affected,
        }

    def wastage_summary(self) -> Dict:
        """
        Summarise units wasted and rejection events.

        Returns
        -------
        dict: {total_wasted, units_expired, units_rejected_quality, wastage_cost}
        """
        df = self._events_df()
        if df.empty:
            return {"total_wasted": 0, "units_expired": 0, "units_rejected_quality": 0}

        waste_df  = df[df["event_type"] == EventType.EXPIRY_WASTE]
        reject_df = df[df["event_type"] == EventType.ORDER_REJECTED]

        units_expired  = int(waste_df.get("units_wasted", pd.Series(dtype=float)).sum())
        units_rejected = int(reject_df.get("order_quantity", pd.Series(dtype=float)).sum())

        return {
            "total_wasted":           units_expired + units_rejected,
            "units_expired":          units_expired,
            "units_rejected_quality": units_rejected,
            "wastage_cost":           round(
                (units_expired + units_rejected) * self.costs["wastage_per_unit"], 2
            ),
        }

    def compliance_rate(self) -> float:
        """
        Cold chain compliance rate = fraction of transits with NO critical excursion.

        Returns
        -------
        float ∈ [0, 1]
        """
        if not self.transit_log:
            return 1.0
        n_compliant = sum(1 for t in self.transit_log if not t["had_critical_excursion"])
        return round(n_compliant / len(self.transit_log), 4)

    def avg_residual_shelf_life(self) -> float:
        """
        Average residual shelf life (days) of units at the point of issue.

        Returns
        -------
        float — days
        """
        if not self.rem_shelf_life_at_issue:
            return 0.0
        return round(float(np.mean([v for _, v in self.rem_shelf_life_at_issue])), 2)

    def emergency_resupply_count(self) -> int:
        """Count of emergency resupply events triggered post-warmup."""
        df = self._events_df()
        if df.empty:
            return 0
        return int((df["event_type"] == EventType.EMERGENCY_RESUPPLY).sum())

    # ─────────────────────────────────────────────────────────────────────────
    # Full summary
    # ─────────────────────────────────────────────────────────────────────────

    def summary(self) -> Dict:
        """
        Return comprehensive KPI summary dict.

        Called after simulation completes (before combining replications).
        """
        cost_bd   = self.total_cost_breakdown()
        exc_sum   = self.excursion_summary()
        waste_sum = self.wastage_summary()

        return {
            "service_level":          round(self.service_level(), 4),
            "service_level_hospital": round(self.service_level(node_id=3), 4),
            "total_cost":             cost_bd["total"],
            "cost_breakdown":         cost_bd,
            "excursions":             exc_sum,
            "wastage":                waste_sum,
            "compliance_rate":        self.compliance_rate(),
            "avg_residual_shelf_life_days": self.avg_residual_shelf_life(),
            "emergency_resupply_count":     self.emergency_resupply_count(),
            "n_events_logged":        len(self.event_log),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Inventory time-series plot
    # ─────────────────────────────────────────────────────────────────────────

    def plot_inventory_timeseries(
        self,
        node_id: int = 3,
        product: str = "PackedRBC",
        rop: Optional[float] = None,
        title: Optional[str] = None,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """
        Plot inventory level over time for a node/product combination.

        Shows reorder point (ROP) as a horizontal dashed line.
        Stockout events are highlighted in red.
        """
        series = self.inv_series.get((node_id, product), [])
        if not series:
            fig, ax_ = plt.subplots(figsize=(12, 4))
            ax_.text(0.5, 0.5, "No inventory data", ha="center", transform=ax_.transAxes)
            return fig

        times = np.array([s[0] for s in series]) / 24.0   # convert hours → days
        units = np.array([s[1] for s in series])
        warmup_days = self.warmup_hours / 24.0

        fig, ax_ = (plt.subplots(figsize=(14, 5)) if ax is None else (ax.get_figure(), ax))
        ax_ = fig.axes[0] if ax is None else ax

        # Shade warmup region
        ax_.axvspan(0, warmup_days, alpha=0.08, color="gray", label="Warm-up period")

        ax_.step(times, units, where="post", color="#2c7bb6", lw=1.2,
                 label=f"Inventory — Node {node_id} ({product})")
        ax_.fill_between(times, units, step="post", alpha=0.15, color="#2c7bb6")

        if rop is not None:
            ax_.axhline(rop, color="#d7191c", lw=1.5, linestyle="--",
                        label=f"Reorder Point (R={rop:.0f})")

        # Shade stockout periods (inventory = 0)
        for i in range(len(units) - 1):
            if units[i] == 0:
                ax_.axvspan(times[i], times[i + 1], alpha=0.25, color="red")

        ax_.set_xlabel("Simulation Day")
        ax_.set_ylabel("Units in Stock")
        ax_.set_title(title or f"Inventory Level — Node {node_id} ({product})")
        ax_.legend(fontsize=8)
        ax_.grid(alpha=0.3)
        plt.tight_layout()
        return fig

    def plot_cost_breakdown(self, ax: Optional[plt.Axes] = None) -> plt.Figure:
        """Stacked bar chart of total cost by category."""
        cost_bd = self.total_cost_breakdown()
        categories = ["holding", "ordering", "shortage", "wastage", "emergency", "maintenance"]
        values     = [cost_bd.get(c, 0.0) for c in categories]
        colors     = ["#4575b4", "#91bfdb", "#d7191c", "#fdae61", "#a50026", "#74add1"]

        fig, ax_ = (plt.subplots(figsize=(8, 5)) if ax is None else (ax.get_figure(), ax))
        ax_ = fig.axes[0] if ax is None else ax

        bars = ax_.bar(categories, values, color=colors, edgecolor="black", alpha=0.85)
        ax_.bar_label(bars, fmt="₹{:.0f}", padding=3, fontsize=8)
        ax_.set_ylabel("Total Cost (₹)")
        ax_.set_title("Cold Chain Cost Breakdown")
        ax_.set_xticklabels([c.replace("_", "\n") for c in categories], rotation=0)
        ax_.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        return fig


# ─────────────────────────────────────────────────────────────────────────────
# Multi-replication aggregation
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_replications(summaries: List[Dict]) -> Dict:
    """
    Aggregate KPI summaries across multiple simulation replications.

    Computes mean ± std for all scalar KPI fields.
    Used by ColdChainSimulation.run_simulation() after n_runs replications.

    Parameters
    ----------
    summaries : list of dicts from MetricsCollector.summary()

    Returns
    -------
    dict: {kpi_name: {mean, std, ci_95_lo, ci_95_hi}}
    """
    if not summaries:
        return {}

    scalar_keys = [
        "service_level",
        "service_level_hospital",
        "total_cost",
        "compliance_rate",
        "avg_residual_shelf_life_days",
        "emergency_resupply_count",
    ]

    result = {}
    n = len(summaries)

    for key in scalar_keys:
        vals = [s.get(key, np.nan) for s in summaries]
        vals = [v for v in vals if not np.isnan(v)]
        if not vals:
            continue
        mu  = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        # 95% CI: mean ± 1.96 × std/√n
        ci  = 1.96 * std / np.sqrt(max(len(vals), 1))
        result[key] = {
            "mean":     round(mu, 4),
            "std":      round(std, 4),
            "ci_95_lo": round(mu - ci, 4),
            "ci_95_hi": round(mu + ci, 4),
            "n_runs":   n,
        }

    # Aggregate cost breakdown
    cost_keys = ["holding", "ordering", "shortage", "wastage", "emergency", "maintenance", "total"]
    cost_agg  = {}
    for ck in cost_keys:
        vals = [s.get("cost_breakdown", {}).get(ck, np.nan) for s in summaries]
        vals = [v for v in vals if not np.isnan(v)]
        if vals:
            cost_agg[ck] = {"mean": round(np.mean(vals), 2), "std": round(np.std(vals, ddof=1) if len(vals)>1 else 0, 2)}
    result["cost_breakdown"] = cost_agg

    # Aggregate excursion/wastage totals
    for sub_key, sub_fields in [
        ("excursions", ["n_warning", "n_critical", "total_duration_min"]),
        ("wastage",    ["total_wasted", "units_expired", "units_rejected_quality"]),
    ]:
        sub_agg = {}
        for f in sub_fields:
            vals = [s.get(sub_key, {}).get(f, np.nan) for s in summaries]
            vals = [v for v in vals if not np.isnan(v)]
            if vals:
                sub_agg[f] = {"mean": round(np.mean(vals), 2), "std": round(np.std(vals, ddof=1) if len(vals)>1 else 0, 2)}
        result[sub_key] = sub_agg

    return result
