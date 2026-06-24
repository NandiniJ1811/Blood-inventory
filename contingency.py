"""
models/contingency.py
=====================
Decision-theoretic contingency protocol for cold chain disruptions.

Formulated as a finite-horizon stochastic dynamic program (Bellman 1957).

State space:
    s = (inventory_level, excursion_detected, time_to_next_replenishment)

Actions when disruption detected:
    A: Emergency resupply from adjacent node
    B: Lateral transfer from sister warehouse
    C: Defer to next scheduled replenishment

Value function (Bellman equation):
    V*(s, t) = min_{a ∈ A} { C(s, a) + E[V*(s', t-1) | s, a] }

Comment: "This follows the structure of a stochastic dynamic program
(Bellman 1957). The state space is kept compact for tractability — in
practice this would use approximate DP or reinforcement learning."

Reference:
    Bellman, R.E. (1957). "Dynamic Programming." Princeton University Press.
    Puterman, M.L. (1994). "Markov Decision Processes." Wiley.

Author: IIT Kharagpur, Dept. of Industrial and Systems Engineering
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, NamedTuple
from itertools import product as iterproduct
from dataclasses import dataclass

DEFAULT_SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# State representation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ContingencyState:
    """
    State in the contingency DP.

    Fields
    ------
    inventory_level      : discretised inventory (0=empty, 1=low, 2=medium, 3=adequate)
    excursion_detected   : 0 = no excursion, 1 = warning, 2 = critical excursion
    hours_to_replenishment: discretised time to next scheduled order (0=imminent, 1=4h, 2=12h, 3=24h+)
    """
    inventory_level:         int   # 0, 1, 2, 3
    excursion_detected:      int   # 0, 1, 2
    hours_to_replenishment:  int   # 0, 1, 2, 3  (index into REPLENISHMENT_HOURS)

    def to_tuple(self) -> Tuple[int, int, int]:
        return (self.inventory_level, self.excursion_detected, self.hours_to_replenishment)


# Discretisation grids
INVENTORY_LEVELS  = [0, 1, 2, 3]             # empty, low, medium, adequate
INV_THRESHOLDS    = [0, 10, 30, 60]          # units (< threshold → that level)
EXCURSION_STATES  = [0, 1, 2]               # none, warning, critical
REPLENISHMENT_HRS = [0, 4, 12, 24]          # hours to next scheduled replenishment

# Actions
ACTION_A = "A"   # Emergency resupply
ACTION_B = "B"   # Lateral transfer
ACTION_C = "C"   # Defer

ACTIONS = [ACTION_A, ACTION_B, ACTION_C]


# ─────────────────────────────────────────────────────────────────────────────
# Cost parameters (₹ per unit; realistic for Indian blood banking)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_PARAMS = {
    # Action A: Emergency resupply
    "emergency_premium":        200.0,   # ₹/unit premium over normal cost
    "expedited_transport_cost": 5000.0,  # ₹ per emergency trip (fixed)
    "emergency_lead_time_h":    4.0,     # mean hours to arrive (Exp distribution)

    # Action B: Lateral transfer
    "transfer_cost_per_unit":   100.0,   # ₹/unit
    "lateral_lead_time_h":      8.0,     # mean hours

    # Action C: Defer
    "shortage_penalty_per_unit_h": 80.0, # ₹/(unit short × hour waited)
    # Expected shortage while deferring:
    # = demand_rate × hours_waited × P(stockout) per unit time

    # General
    "demand_rate_per_h":        1.04,    # units/hour = 25/day
    "units_per_order":          50,      # standard order size
    "unit_cost":                1500.0,  # ₹/unit (Packed RBC)
}


# ─────────────────────────────────────────────────────────────────────────────
# Contingency Protocol DP
# ─────────────────────────────────────────────────────────────────────────────

class ContingencyProtocol:
    """
    Stochastic DP model for contingency action selection under cold chain disruptions.

    State: (inventory_level, excursion_detected, hours_to_replenishment)
    Actions: A (emergency), B (lateral), C (defer)

    Value function computed by backward induction over finite horizon.

    Comment: "This follows the structure of a stochastic dynamic program
    (Bellman 1957). The state space is kept compact for tractability —
    3 × 3 × 4 = 36 states × 3 actions. In practice this would use
    approximate DP or reinforcement learning for larger state spaces."
    """

    def __init__(
        self,
        params: Optional[Dict] = None,
        horizon_hours: int = 24,
        seed: int = DEFAULT_SEED,
    ):
        """
        Parameters
        ----------
        params        : cost / lead time parameters (defaults to DEFAULT_PARAMS)
        horizon_hours : planning horizon for DP (hours)
        seed          : random seed
        """
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.horizon = horizon_hours
        self.rng     = np.random.default_rng(seed)

        # State space
        self.states = [
            ContingencyState(inv, exc, rep)
            for inv in INVENTORY_LEVELS
            for exc in EXCURSION_STATES
            for rep in range(len(REPLENISHMENT_HRS))
        ]

        # Value table: V[state_tuple][t]
        self._V: Dict[Tuple, np.ndarray] = {}
        self._policy: Dict[Tuple, np.ndarray] = {}
        self._solved = False

    # ── Immediate cost of taking action a in state s ──────────────────────────

    def immediate_cost(self, state: ContingencyState, action: str) -> float:
        """
        Compute expected immediate cost of action in current state.

        Costs are based on:
            A: premium + fixed transport + expected units needed × unit cost
            B: transfer cost × units needed
            C: shortage penalty × expected demand × hours to replenishment

        Parameters
        ----------
        state  : current state
        action : 'A', 'B', or 'C'

        Returns
        -------
        float — expected immediate cost (₹)
        """
        p = self.params
        inv_level = state.inventory_level

        # Units shortage estimate based on inventory level and demand during lead time
        hours_to_next = REPLENISHMENT_HRS[state.hours_to_replenishment]

        # Expected demand during lead time (assuming Poisson demand)
        expected_demand_emergency = p["demand_rate_per_h"] * p["emergency_lead_time_h"]
        expected_demand_lateral   = p["demand_rate_per_h"] * p["lateral_lead_time_h"]
        expected_demand_defer     = p["demand_rate_per_h"] * hours_to_next

        # Inventory buffer (units in stock)
        inv_buffer = INV_THRESHOLDS[inv_level]

        if action == ACTION_A:
            # Emergency resupply: pay premium + transport, get units in ~4h
            units_needed = max(0, expected_demand_emergency - inv_buffer)
            cost = (units_needed * p["emergency_premium"]
                    + p["expedited_transport_cost"]
                    + units_needed * p["unit_cost"])
            # If critical excursion, also lose current inventory (wastage)
            if state.excursion_detected == 2:
                cost += inv_buffer * p["unit_cost"] * 0.5   # 50% expected loss from excursion

        elif action == ACTION_B:
            # Lateral transfer: cheaper per unit but longer lead time
            units_needed = max(0, expected_demand_lateral - inv_buffer)
            cost = units_needed * p["transfer_cost_per_unit"]
            # Shortage penalty during 8h wait
            shortfall = max(0, expected_demand_lateral - inv_buffer)
            cost += shortfall * p["shortage_penalty_per_unit_h"] * p["lateral_lead_time_h"]

        elif action == ACTION_C:
            # Defer: wait for next scheduled replenishment
            shortfall = max(0, expected_demand_defer - inv_buffer)
            cost = (shortfall * p["shortage_penalty_per_unit_h"] * hours_to_next
                    + shortfall * p["unit_cost"] * 0.2)   # 20% lost goodwill premium
            # Higher cost if excursion detected (product may be compromised)
            if state.excursion_detected == 2:
                cost *= 1.5

        else:
            raise ValueError(f"Unknown action: {action}")

        return max(0.0, round(cost, 2))

    # ── State transition (simplified stochastic model) ────────────────────────

    def transition_probs(
        self,
        state: ContingencyState,
        action: str,
    ) -> List[Tuple[ContingencyState, float]]:
        """
        Simplified state transition probabilities after taking action.

        Returns list of (next_state, probability) pairs.
        Probabilities are heuristic approximations based on action outcomes.
        """
        next_states = []
        inv  = state.inventory_level
        exc  = state.excursion_detected
        rep  = state.hours_to_replenishment

        if action == ACTION_A:
            # Emergency resupply → likely restores adequate inventory
            # Excursion cleared after resupply (fresh product)
            new_inv = min(3, inv + 2)
            new_exc = 0
            new_rep = min(3, rep + 1)   # push out next scheduled order slightly
            next_states.append((ContingencyState(new_inv, new_exc, new_rep), 0.85))
            next_states.append((ContingencyState(max(0, new_inv - 1), 0, new_rep), 0.15))

        elif action == ACTION_B:
            # Lateral transfer → partial improvement, longer lead time
            new_inv = min(3, inv + 1)
            new_exc = max(0, exc - 1)  # excursion may persist
            new_rep = rep
            next_states.append((ContingencyState(new_inv, new_exc, new_rep), 0.70))
            next_states.append((ContingencyState(inv, exc, new_rep), 0.30))

        elif action == ACTION_C:
            # Defer → inventory decreases with demand
            new_inv = max(0, inv - 1)
            # Excursion may escalate or resolve
            if exc == 0:
                next_states.append((ContingencyState(new_inv, 0, max(0, rep - 1)), 0.85))
                next_states.append((ContingencyState(new_inv, 1, max(0, rep - 1)), 0.15))
            elif exc == 1:
                next_states.append((ContingencyState(new_inv, 1, max(0, rep - 1)), 0.5))
                next_states.append((ContingencyState(new_inv, 2, max(0, rep - 1)), 0.3))
                next_states.append((ContingencyState(new_inv, 0, max(0, rep - 1)), 0.2))
            else:  # critical
                next_states.append((ContingencyState(max(0, new_inv - 1), 2, max(0, rep - 1)), 0.6))
                next_states.append((ContingencyState(new_inv, 1, max(0, rep - 1)), 0.4))

        return next_states

    # ── Backward Induction (value iteration) ─────────────────────────────────

    def value_function(self, horizon: Optional[int] = None) -> Dict:
        """
        Compute optimal value function V*(s) via backward induction.

        Bellman recursion:
            V*(s, t) = min_{a} { C(s, a) + γ × Σ_{s'} P(s'|s,a) × V*(s', t-1) }
        Terminal condition: V*(s, 0) = 0 for all s (no future cost at horizon end)

        Reference: Bellman, R.E. (1957). "Dynamic Programming." Princeton UP.

        Parameters
        ----------
        horizon : planning horizon in steps (each step = 1 hour)

        Returns
        -------
        dict: {V_star, optimal_policy}
        """
        H = horizon or self.horizon
        gamma = 0.99  # discount factor (close to 1 for 24h horizon)

        # Initialise V[s] = 0 for all states (terminal value)
        V = {s.to_tuple(): 0.0 for s in self.states}
        policy = {s.to_tuple(): ACTION_C for s in self.states}

        # Backward induction over H steps
        for t in range(1, H + 1):
            V_new = {}
            policy_new = {}
            for s in self.states:
                # Compute Q-value for each action
                q_values = {}
                for a in ACTIONS:
                    immediate = self.immediate_cost(s, a)
                    trans = self.transition_probs(s, a)
                    expected_future = sum(prob * V[s_next.to_tuple()] for s_next, prob in trans)
                    q_values[a] = immediate + gamma * expected_future

                # Optimal action minimises Q-value
                best_action = min(q_values, key=q_values.get)
                V_new[s.to_tuple()]      = q_values[best_action]
                policy_new[s.to_tuple()] = best_action

            V      = V_new
            policy = policy_new

        self._V      = V
        self._policy = policy
        self._solved = True

        # Format output
        return {
            "V_star":         {str(k): round(v, 2) for k, v in V.items()},
            "optimal_policy": {str(k): v for k, v in policy.items()},
            "horizon":        H,
        }

    # ── Optimal action query ──────────────────────────────────────────────────

    def optimal_action(
        self,
        inventory_units: float,
        excursion_status: int,
        hours_to_replenishment: float,
    ) -> Dict:
        """
        Query the optimal action for a given operational state.

        Parameters
        ----------
        inventory_units         : current stock level (units)
        excursion_status        : 0=none, 1=warning, 2=critical
        hours_to_replenishment  : hours until next scheduled replenishment

        Returns
        -------
        dict: {action, action_name, expected_cost, reasoning}
        """
        if not self._solved:
            self.value_function()

        # Discretise continuous state
        inv_idx = sum(1 for t in INV_THRESHOLDS if inventory_units > t) - 1
        inv_idx = max(0, min(3, inv_idx))

        rep_idx = 0
        for i, h in enumerate(REPLENISHMENT_HRS):
            if hours_to_replenishment >= h:
                rep_idx = i

        state = ContingencyState(inv_idx, excursion_status, rep_idx)
        s_key = state.to_tuple()

        action = self._policy.get(s_key, ACTION_C)
        cost   = self._V.get(s_key, 0.0)

        # Compute cost for all actions (for transparency)
        action_costs = {a: self.immediate_cost(state, a) for a in ACTIONS}

        action_names = {
            ACTION_A: "Emergency Resupply",
            ACTION_B: "Lateral Transfer",
            ACTION_C: "Defer to Scheduled",
        }
        reasoning = {
            ACTION_A: (
                "Critical situation: stockout risk is high and replenishment is far away. "
                "Emergency resupply minimises expected shortage penalty despite premium cost."
            ),
            ACTION_B: (
                "Moderate situation: lateral transfer provides cost-effective coverage "
                "while preserving emergency response capacity."
            ),
            ACTION_C: (
                "Low urgency: current inventory is adequate until next scheduled replenishment. "
                "Deferral minimises total cost."
            ),
        }

        return {
            "action":           action,
            "action_name":      action_names[action],
            "expected_cost_dp": round(cost, 2),
            "immediate_costs":  {k: round(v, 2) for k, v in action_costs.items()},
            "state_discretised": {
                "inventory_level": inv_idx,
                "excursion_status": excursion_status,
                "hours_to_replenishment_idx": rep_idx,
            },
            "reasoning":        reasoning[action],
        }

    # ── Cost comparison plot ──────────────────────────────────────────────────

    def cost_comparison_plot(
        self,
        inventory_units: float,
        excursion_status: int,
        hours_to_replenishment: float,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """
        Bar chart: expected cost under each action for the current state.

        Parameters
        ----------
        inventory_units, excursion_status, hours_to_replenishment : current state
        """
        info = self.optimal_action(inventory_units, excursion_status, hours_to_replenishment)

        actions = list(info["immediate_costs"].keys())
        costs   = [info["immediate_costs"][a] for a in actions]
        labels  = ["A: Emergency\nResupply", "B: Lateral\nTransfer", "C: Defer"]
        colors  = ["#d7191c", "#fdae61", "#1a9641"]

        fig, ax_ = (plt.subplots(figsize=(8, 5)) if ax is None else (ax.get_figure(), ax))
        ax_ = fig.axes[0] if ax is None else ax

        bars = ax_.bar(labels, costs, color=colors, alpha=0.85, edgecolor="black")

        # Highlight optimal action
        opt_idx = actions.index(info["action"])
        bars[opt_idx].set_edgecolor("black")
        bars[opt_idx].set_linewidth(3)

        ax_.text(
            opt_idx, costs[opt_idx] + max(costs) * 0.02,
            "★ OPTIMAL", ha="center", fontsize=9, color="black", fontweight="bold",
        )
        ax_.set_ylabel("Expected Cost (₹)")
        ax_.set_title(
            f"Contingency Action Cost Comparison\n"
            f"State: Inventory={inventory_units:.0f}u, "
            f"Excursion={['None','Warning','Critical'][excursion_status]}, "
            f"Next reorder in {hours_to_replenishment:.0f}h"
        )
        ax_.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        return fig

    # ── Sensitivity heatmap ───────────────────────────────────────────────────

    def sensitivity_to_parameters(
        self,
        shortage_penalties: Optional[np.ndarray] = None,
        replenishment_hours_range: Optional[np.ndarray] = None,
        inventory_units: float = 15.0,
        excursion_status: int = 2,
    ) -> Tuple[plt.Figure, pd.DataFrame]:
        """
        2D heatmap: shortage_penalty (y) × hours_to_next_reorder (x) → optimal action.

        Shows how optimal action shifts from Defer → Lateral → Emergency
        as shortage penalty increases or time to replenishment grows.

        Parameters
        ----------
        shortage_penalties       : array of p values to sweep
        replenishment_hours_range: array of hours_to_next to sweep
        inventory_units          : fixed inventory level for this analysis
        excursion_status         : fixed excursion status (0, 1, or 2)

        Returns
        -------
        (Figure, DataFrame of optimal actions)
        """
        if shortage_penalties is None:
            shortage_penalties = np.linspace(10, 200, 10)
        if replenishment_hours_range is None:
            replenishment_hours_range = np.array([1, 4, 8, 12, 18, 24, 36, 48])

        action_map = {ACTION_A: 2, ACTION_B: 1, ACTION_C: 0}
        action_labels = {2: "A: Emergency", 1: "B: Lateral", 0: "C: Defer"}

        grid = np.zeros((len(shortage_penalties), len(replenishment_hours_range)), dtype=int)
        records = []

        for i, sp in enumerate(shortage_penalties):
            for j, h in enumerate(replenishment_hours_range):
                # Re-run DP with this shortage penalty
                self.params["shortage_penalty_per_unit_h"] = sp
                self.value_function()
                result = self.optimal_action(inventory_units, excursion_status, h)
                grid[i, j] = action_map[result["action"]]
                records.append({
                    "shortage_penalty": round(sp, 1),
                    "hours_to_replenishment": h,
                    "optimal_action": result["action"],
                    "action_name": result["action_name"],
                })

        # Restore default
        self.params["shortage_penalty_per_unit_h"] = DEFAULT_PARAMS["shortage_penalty_per_unit_h"]

        df = pd.DataFrame(records)

        fig, ax_ = plt.subplots(figsize=(10, 6))
        cmap = plt.cm.get_cmap("RdYlGn", 3)
        im = ax_.imshow(grid, aspect="auto", cmap=cmap, vmin=-0.5, vmax=2.5, origin="lower")

        ax_.set_xticks(range(len(replenishment_hours_range)))
        ax_.set_xticklabels([f"{h}h" for h in replenishment_hours_range])
        ax_.set_yticks(range(len(shortage_penalties)))
        ax_.set_yticklabels([f"₹{sp:.0f}" for sp in shortage_penalties])
        ax_.set_xlabel("Hours to Next Scheduled Replenishment")
        ax_.set_ylabel("Shortage Penalty (₹/unit/hour)")
        ax_.set_title(
            f"Optimal Contingency Action\n"
            f"Inventory={inventory_units:.0f}u, "
            f"Excursion={'None/Warning/Critical'[excursion_status]}\n"
            f"Green=Defer, Yellow=Lateral, Red=Emergency"
        )

        cbar = plt.colorbar(im, ax=ax_, ticks=[0, 1, 2])
        cbar.ax.set_yticklabels(["C: Defer", "B: Lateral", "A: Emergency"])

        plt.tight_layout()
        return fig, df
