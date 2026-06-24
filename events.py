"""
simulation/events.py
====================
SimPy event definitions, named tuples, and event constants for the
cold chain discrete-event simulation.

All simulation events are logged with timestamps and node identifiers
so they can be aggregated into KPIs by the MetricsCollector.

Author: IIT Kharagpur, Dept. of Industrial and Systems Engineering
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Event type constants
# ─────────────────────────────────────────────────────────────────────────────

class EventType:
    """String constants for event type classification."""
    DEMAND_FULFILLED      = "demand_fulfilled"
    DEMAND_PARTIAL        = "demand_partial"
    STOCKOUT              = "stockout"
    ORDER_PLACED          = "order_placed"
    ORDER_ARRIVED         = "order_arrived"
    ORDER_REJECTED        = "order_rejected_quality"
    EXPIRY_WASTE          = "expiry_waste"
    TEMP_EXCURSION_WARN   = "temp_excursion_warning"
    TEMP_EXCURSION_CRIT   = "temp_excursion_critical"
    TEMP_EXCURSION_END    = "temp_excursion_end"
    EQUIPMENT_FAILURE     = "equipment_failure"
    EQUIPMENT_REPAIRED    = "equipment_repaired"
    EMERGENCY_RESUPPLY    = "emergency_resupply"
    LATERAL_TRANSFER      = "lateral_transfer"
    REPLENISHMENT_DEFERRED= "replenishment_deferred"
    SIMULATION_START      = "simulation_start"
    SIMULATION_END        = "simulation_end"
    WARMUP_END            = "warmup_end"


# ─────────────────────────────────────────────────────────────────────────────
# Event data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SimEvent:
    """
    Base simulation event record.

    Every event in the simulation is logged as a SimEvent (or subclass)
    and stored in the MetricsCollector event log.
    """
    time:       float          # simulation time (hours)
    event_type: str            # EventType constant
    node_id:    int            # which node
    product:    str            # product type
    details:    Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "time":       self.time,
            "event_type": self.event_type,
            "node_id":    self.node_id,
            "product":    self.product,
            **self.details,
        }


@dataclass
class DemandEvent(SimEvent):
    """Records a demand arrival and fulfillment outcome."""
    demand_quantity:   int   = 0
    fulfilled_quantity: int  = 0
    shortage:          int   = 0

    def __post_init__(self):
        self.event_type = (
            EventType.DEMAND_FULFILLED if self.shortage == 0
            else (EventType.DEMAND_PARTIAL if self.fulfilled_quantity > 0
                  else EventType.STOCKOUT)
        )
        self.details.update({
            "demand":    self.demand_quantity,
            "fulfilled": self.fulfilled_quantity,
            "shortage":  self.shortage,
        })


@dataclass
class OrderEvent(SimEvent):
    """Records an order placed, arrived, or rejected."""
    order_quantity:  int   = 0
    quality_index:   float = 1.0
    accepted:        bool  = True
    lead_time_h:     float = 0.0

    def __post_init__(self):
        if self.event_type == EventType.ORDER_ARRIVED and not self.accepted:
            self.event_type = EventType.ORDER_REJECTED
        self.details.update({
            "order_quantity": self.order_quantity,
            "quality_index":  round(self.quality_index, 4),
            "accepted":       self.accepted,
            "lead_time_h":    round(self.lead_time_h, 2),
        })


@dataclass
class ExcursionEvent(SimEvent):
    """Records a temperature excursion start or end."""
    max_temp:          float = 0.0
    min_temp:          float = 0.0
    duration_minutes:  float = 0.0
    zone:              str   = "warning"

    def __post_init__(self):
        self.details.update({
            "max_temp":         self.max_temp,
            "min_temp":         self.min_temp,
            "duration_minutes": self.duration_minutes,
            "zone":             self.zone,
        })


@dataclass
class WastageEvent(SimEvent):
    """Records units wasted due to expiry or quality rejection."""
    units_wasted:   int   = 0
    reason:         str   = "expiry"   # 'expiry' or 'quality_rejection'
    wastage_cost:   float = 0.0

    def __post_init__(self):
        self.event_type = EventType.EXPIRY_WASTE
        self.details.update({
            "units_wasted": self.units_wasted,
            "reason":       self.reason,
            "wastage_cost": round(self.wastage_cost, 2),
        })


@dataclass
class EquipmentEvent(SimEvent):
    """Records equipment failure or repair."""
    repair_duration_h: float = 0.0
    downtime_h:        float = 0.0

    def __post_init__(self):
        self.details.update({
            "repair_duration_h": round(self.repair_duration_h, 2),
            "downtime_h":        round(self.downtime_h, 2),
        })


@dataclass
class ContingencyActionEvent(SimEvent):
    """Records a contingency action taken (emergency / lateral / defer)."""
    action:        str   = "C"
    units_ordered: int   = 0
    action_cost:   float = 0.0
    trigger:       str   = "excursion"   # 'excursion' or 'stockout'

    def __post_init__(self):
        self.event_type = {
            "A": EventType.EMERGENCY_RESUPPLY,
            "B": EventType.LATERAL_TRANSFER,
            "C": EventType.REPLENISHMENT_DEFERRED,
        }.get(self.action, EventType.REPLENISHMENT_DEFERRED)
        self.details.update({
            "action":        self.action,
            "units_ordered": self.units_ordered,
            "action_cost":   round(self.action_cost, 2),
            "trigger":       self.trigger,
        })


# ─────────────────────────────────────────────────────────────────────────────
# SimEvents namespace — factory methods for creating events
# ─────────────────────────────────────────────────────────────────────────────

class SimEvents:
    """
    Factory namespace for creating typed simulation events.

    Usage:
        event = SimEvents.demand(time=10.5, node_id=3, product='PackedRBC',
                                 demand=25, fulfilled=25, shortage=0)
    """

    @staticmethod
    def demand(time, node_id, product, demand, fulfilled, shortage) -> DemandEvent:
        return DemandEvent(
            time=time, event_type="", node_id=node_id, product=product,
            demand_quantity=demand, fulfilled_quantity=fulfilled, shortage=shortage,
        )

    @staticmethod
    def order_placed(time, node_id, product, quantity) -> OrderEvent:
        return OrderEvent(
            time=time, event_type=EventType.ORDER_PLACED,
            node_id=node_id, product=product, order_quantity=quantity,
        )

    @staticmethod
    def order_arrived(time, node_id, product, quantity, quality_index, lead_time_h, accepted) -> OrderEvent:
        return OrderEvent(
            time=time, event_type=EventType.ORDER_ARRIVED,
            node_id=node_id, product=product, order_quantity=quantity,
            quality_index=quality_index, lead_time_h=lead_time_h, accepted=accepted,
        )

    @staticmethod
    def excursion(time, node_id, product, max_temp, min_temp, zone, duration_minutes=0.0) -> ExcursionEvent:
        etype = EventType.TEMP_EXCURSION_CRIT if zone == "critical" else EventType.TEMP_EXCURSION_WARN
        return ExcursionEvent(
            time=time, event_type=etype, node_id=node_id, product=product,
            max_temp=max_temp, min_temp=min_temp, zone=zone, duration_minutes=duration_minutes,
        )

    @staticmethod
    def wastage(time, node_id, product, units_wasted, reason, wastage_cost) -> WastageEvent:
        return WastageEvent(
            time=time, event_type="", node_id=node_id, product=product,
            units_wasted=units_wasted, reason=reason, wastage_cost=wastage_cost,
        )

    @staticmethod
    def equipment_failure(time, node_id, product, repair_duration_h) -> EquipmentEvent:
        return EquipmentEvent(
            time=time, event_type=EventType.EQUIPMENT_FAILURE,
            node_id=node_id, product=product, repair_duration_h=repair_duration_h,
        )

    @staticmethod
    def equipment_repaired(time, node_id, product, downtime_h) -> EquipmentEvent:
        return EquipmentEvent(
            time=time, event_type=EventType.EQUIPMENT_REPAIRED,
            node_id=node_id, product=product, downtime_h=downtime_h,
        )

    @staticmethod
    def contingency(time, node_id, product, action, units_ordered, action_cost, trigger) -> ContingencyActionEvent:
        return ContingencyActionEvent(
            time=time, event_type="", node_id=node_id, product=product,
            action=action, units_ordered=units_ordered,
            action_cost=action_cost, trigger=trigger,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Demand seasonality helpers
# ─────────────────────────────────────────────────────────────────────────────

def seasonal_demand_multiplier(sim_day: float) -> float:
    """
    Returns a multiplicative seasonal factor for demand.

    Indian context:
      - Higher demand May–September (summer: accidents, heat-related illness)
      - Mid-week peak (Tuesday–Thursday)
      - Occasional Poisson spikes (mass casualty events)

    Parameters
    ----------
    sim_day : simulation day (day 0 = Jan 1 equivalent)

    Returns
    -------
    float — multiplier (≥ 0.7, typical range 0.85–1.35)
    """
    day_of_year = sim_day % 365
    day_of_week = int(sim_day % 7)

    # Monthly seasonal factor: higher May (day ~120) – Sep (day ~273)
    # Smooth sinusoidal approximation with peak in July (day 196)
    seasonal = 1.0 + 0.20 * np.sin(2 * np.pi * (day_of_year - 105) / 365)

    # Weekly pattern: 1.10 mid-week (Tue/Wed/Thu), 0.85 weekend
    weekly_factors = [0.90, 1.05, 1.12, 1.10, 1.05, 0.85, 0.88]   # Mon–Sun
    weekly = weekly_factors[day_of_week % 7]

    return round(seasonal * weekly, 4)


def sample_daily_demand(
    sim_day: float,
    base_mean: float = 25.0,
    base_std: float = 7.0,
    accident_rate: float = 0.02,   # Poisson rate: ~1 spike per 50 days
    accident_extra_units: float = 40.0,
    rng: Optional[np.random.Generator] = None,
) -> int:
    """
    Sample stochastic daily demand for hospital (Node 3).

    Components:
        1. Base demand: N(μ, σ²) × seasonal_multiplier
        2. Accident surge: Poisson(λ) events, each adding extra units

    Parameters
    ----------
    sim_day              : current simulation day
    base_mean            : μ — average daily demand
    base_std             : σ — std of daily demand
    accident_rate        : λ — Poisson rate of mass-casualty surge events per day
    accident_extra_units : mean extra units per surge event
    rng                  : numpy random Generator (optional)

    Returns
    -------
    int — sampled demand for the day (non-negative)
    """
    if rng is None:
        rng = np.random.default_rng()

    multiplier = seasonal_demand_multiplier(sim_day)
    mu_adj = base_mean * multiplier

    # Base demand ~ N(μ_adj, σ²)
    base_demand = rng.normal(mu_adj, base_std)

    # Accident surge: Poisson number of events, each adds Exp(accident_extra_units) units
    n_accidents = rng.poisson(accident_rate)
    surge = sum(rng.exponential(accident_extra_units) for _ in range(n_accidents))

    total = max(0, int(round(base_demand + surge)))
    return total
