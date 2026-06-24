"""
models/network.py
=================
Cold chain network topology model using NetworkX directed graph.

The 4-node network represents a typical Indian blood supply chain:
    Node 0: Blood Bank / Manufacturer (source)
    Node 1: Regional Cold Storage Hub
    Node 2: District Warehouse
    Node 3: Hospital / End-use facility (sink)

Network analysis methods:
  - Shortest path (Dijkstra's algorithm)
  - Criticality via betweenness centrality (Newman 2010)
  - Network visualisation

Key insight: "High betweenness centrality nodes are single points of failure
— disruption here propagates to all downstream nodes." Node 1 (Regional Hub)
typically has the highest betweenness centrality in this linear chain topology.

Reference: Newman, M.E.J. (2010). "Networks: An Introduction." Oxford University Press.
           Chapter 7: Measures and metrics.

Author: IIT Kharagpur, Dept. of Industrial and Systems Engineering
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    raise ImportError("networkx is required: pip install networkx")


# ─────────────────────────────────────────────────────────────────────────────
# Default network configuration (fallback if JSON not found)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_NETWORK_CONFIG = {
    "nodes": [
        {
            "id": 0,
            "name": "Blood Bank (Source)",
            "type": "source",
            "capacity": 2000,
            "temp_target_c": 4.0,
            "temp_variance": 0.5,
            "holding_cost_per_unit_day": 2.0,
            "mttf_days": 365,
            "review_period_days": 1,
            "x": 0.1,  # layout coordinates
            "y": 0.5,
        },
        {
            "id": 1,
            "name": "Regional Cold Storage Hub",
            "type": "hub",
            "capacity": 800,
            "temp_target_c": 4.0,
            "temp_variance": 0.8,
            "holding_cost_per_unit_day": 3.5,
            "mttf_days": 180,
            "review_period_days": 2,
            "x": 0.37,
            "y": 0.5,
        },
        {
            "id": 2,
            "name": "District Warehouse",
            "type": "warehouse",
            "capacity": 400,
            "temp_target_c": 4.0,
            "temp_variance": 1.0,
            "holding_cost_per_unit_day": 4.0,
            "mttf_days": 120,
            "review_period_days": 3,
            "x": 0.63,
            "y": 0.5,
        },
        {
            "id": 3,
            "name": "Hospital (Sink)",
            "type": "sink",
            "capacity": 200,
            "temp_target_c": 4.0,
            "temp_variance": 1.2,
            "holding_cost_per_unit_day": 5.0,
            "mttf_days": 90,
            "review_period_days": 1,
            "x": 0.9,
            "y": 0.5,
        },
    ],
    "edges": [
        {
            "from": 0, "to": 1,
            "transit_time_mean_h": 5.0,
            "transit_time_std_h": 0.8,
            "temp_std_c": 0.6,
            "failure_rate_per_100_trips": 2.0,
            "distance_km": 120.0,
            "cost_per_unit": 8.0,
        },
        {
            "from": 1, "to": 2,
            "transit_time_mean_h": 3.0,
            "transit_time_std_h": 0.6,
            "temp_std_c": 0.9,
            "failure_rate_per_100_trips": 3.5,
            "distance_km": 60.0,
            "cost_per_unit": 5.0,
        },
        {
            "from": 2, "to": 3,
            "transit_time_mean_h": 1.5,
            "transit_time_std_h": 0.3,
            "temp_std_c": 1.1,
            "failure_rate_per_100_trips": 4.0,
            "distance_km": 25.0,
            "cost_per_unit": 3.0,
        },
    ],
}

# Node type → colour mapping for visualisation
NODE_COLORS = {
    "source":    "#2166ac",
    "hub":       "#4dac26",
    "warehouse": "#f4a582",
    "sink":      "#d6604d",
}


class ColdChainNetwork:
    """
    Directed graph model of a cold chain distribution network.

    Wraps NetworkX DiGraph with domain-specific attributes, analysis methods,
    and visualisation capabilities.

    Attributes stored on nodes:
        name, type, capacity, temp_target_c, temp_variance,
        holding_cost_per_unit_day, mttf_days, review_period_days

    Attributes stored on edges:
        transit_time_mean_h, transit_time_std_h, temp_std_c,
        failure_rate_per_100_trips, distance_km, cost_per_unit,
        transit_risk (computed)
    """

    def __init__(self, config: Optional[Dict] = None, config_path: Optional[str] = None):
        """
        Parameters
        ----------
        config      : dict — network configuration (overrides config_path)
        config_path : path to sample_network.json
        """
        if not HAS_NETWORKX:
            raise RuntimeError("networkx is not installed.")

        self.G: nx.DiGraph = nx.DiGraph()
        self._node_map: Dict[int, str] = {}   # node_id → name

        # Load config
        if config is not None:
            self._config = config
        elif config_path is not None:
            with open(config_path, "r") as f:
                self._config = json.load(f)
        else:
            self._config = DEFAULT_NETWORK_CONFIG

        self._build_from_config(self._config)

    # ─────────────────────────────────────────────────────────────────────────
    # Build graph from config
    # ─────────────────────────────────────────────────────────────────────────

    def _build_from_config(self, config: Dict):
        """Populate NetworkX DiGraph from configuration dict."""
        for node_cfg in config["nodes"]:
            nid = node_cfg["id"]
            self.G.add_node(nid, **{k: v for k, v in node_cfg.items() if k != "id"})
            self._node_map[nid] = node_cfg.get("name", f"Node {nid}")

        for edge_cfg in config["edges"]:
            src = edge_cfg["from"]
            dst = edge_cfg["to"]
            attrs = {k: v for k, v in edge_cfg.items() if k not in ("from", "to")}
            # Compute composite transit_risk score for edge weighting
            # transit_risk ∈ [0, 1] — higher = riskier transit
            attrs["transit_risk"] = self._compute_transit_risk(
                attrs["transit_time_mean_h"],
                attrs["temp_std_c"],
                attrs["failure_rate_per_100_trips"],
            )
            attrs["cost"] = attrs["cost_per_unit"]  # alias for Dijkstra weight
            self.G.add_edge(src, dst, **attrs)

    @staticmethod
    def _compute_transit_risk(transit_h: float, temp_std: float, fail_rate: float) -> float:
        """
        Composite transit risk score (heuristic, normalised to [0,1]).

        Components:
            - Time risk: longer transit → more exposure → higher risk
            - Temperature variability risk: higher σ → more excursion risk
            - Vehicle failure risk: higher failure_rate → higher risk

        All components normalised and weighted equally.
        """
        # Normalise against plausible ranges
        time_risk   = min(1.0, transit_h / 24.0)           # 24h = max reference
        temp_risk   = min(1.0, temp_std / 3.0)             # 3°C std = max reference
        fail_risk   = min(1.0, fail_rate / 10.0)           # 10/100 trips = max reference
        return round((time_risk + temp_risk + fail_risk) / 3.0, 4)

    # ─────────────────────────────────────────────────────────────────────────
    # Node / Edge management
    # ─────────────────────────────────────────────────────────────────────────

    def add_node(
        self,
        node_id: int,
        node_type: str,
        capacity: int,
        temp_target: float,
        holding_cost: float,
        name: Optional[str] = None,
        **kwargs,
    ):
        """
        Add or update a network node.

        Parameters
        ----------
        node_id     : integer ID
        node_type   : 'source', 'hub', 'warehouse', 'sink'
        capacity    : max storage capacity (units)
        temp_target : target storage temperature (°C)
        holding_cost: holding cost per unit per day (₹)
        name        : optional human-readable name
        **kwargs    : additional node attributes
        """
        attrs = {
            "type":                  node_type,
            "capacity":              capacity,
            "temp_target_c":         temp_target,
            "holding_cost_per_unit_day": holding_cost,
            "name":                  name or f"Node {node_id}",
            **kwargs,
        }
        self.G.add_node(node_id, **attrs)
        self._node_map[node_id] = attrs["name"]

    def add_link(
        self,
        from_node: int,
        to_node: int,
        transit_time_h: float,
        temp_std: float,
        failure_rate: float,
        distance_km: float,
        unit_cost: float,
        transit_time_std_h: float = 0.5,
    ):
        """
        Add or update a network edge (transit link).

        Parameters
        ----------
        from_node        : source node ID
        to_node          : destination node ID
        transit_time_h   : mean transit time (hours)
        temp_std         : temperature std during transit (°C)
        failure_rate     : vehicle failures per 100 trips
        distance_km      : route distance (km)
        unit_cost        : transport cost per unit (₹)
        transit_time_std_h: std of transit time (hours)
        """
        risk = self._compute_transit_risk(transit_time_h, temp_std, failure_rate)
        self.G.add_edge(from_node, to_node,
                        transit_time_mean_h=transit_time_h,
                        transit_time_std_h=transit_time_std_h,
                        temp_std_c=temp_std,
                        failure_rate_per_100_trips=failure_rate,
                        distance_km=distance_km,
                        cost_per_unit=unit_cost,
                        cost=unit_cost,
                        transit_risk=risk)

    def get_node_attr(self, node_id: int) -> Dict:
        """Return all attributes of a node as dict."""
        return dict(self.G.nodes[node_id])

    def get_edge_attr(self, from_node: int, to_node: int) -> Dict:
        """Return all attributes of an edge as dict."""
        return dict(self.G[from_node][to_node])

    # ─────────────────────────────────────────────────────────────────────────
    # Network analysis
    # ─────────────────────────────────────────────────────────────────────────

    def shortest_path(
        self,
        source: int = 0,
        sink: int = 3,
        weight: str = "cost",
    ) -> Tuple[List[int], float]:
        """
        Compute shortest (cheapest) path using Dijkstra's algorithm.

        Parameters
        ----------
        source : source node ID
        sink   : destination node ID
        weight : edge attribute to minimise ('cost', 'transit_time_mean_h', 'distance_km')

        Returns
        -------
        (path_node_list, total_weight)
        """
        try:
            path = nx.dijkstra_path(self.G, source, sink, weight=weight)
            cost = nx.dijkstra_path_length(self.G, source, sink, weight=weight)
        except nx.NetworkXNoPath:
            return [], np.inf
        return path, round(cost, 4)

    def criticality_scores(self) -> Dict:
        """
        Compute betweenness centrality for nodes and edges.

        Betweenness centrality measures how many shortest paths pass through
        a node/edge. High centrality = high criticality = single point of failure.

        Comment: "High betweenness centrality nodes are single points of
        failure — disruption here propagates to all downstream nodes.
        In this linear 4-node chain, Node 1 (Hub) has the highest centrality
        because ALL flows between source and hospital pass through it."

        Reference: Newman, M.E.J. (2010). "Networks: An Introduction."
                   Oxford University Press. Chapter 7.

        Returns
        -------
        dict: {
            'node_betweenness': {node_id: score},
            'edge_betweenness': {(u,v): score},
            'most_critical_node': node_id,
            'most_critical_edge': (u, v),
        }
        """
        node_bc = nx.betweenness_centrality(self.G, weight="cost", normalized=True)
        edge_bc = nx.edge_betweenness_centrality(self.G, weight="cost", normalized=True)

        most_critical_node = max(node_bc, key=node_bc.get)
        most_critical_edge = max(edge_bc, key=edge_bc.get)

        return {
            "node_betweenness":    {k: round(v, 4) for k, v in node_bc.items()},
            "edge_betweenness":    {str(k): round(v, 4) for k, v in edge_bc.items()},
            "most_critical_node":  most_critical_node,
            "most_critical_node_name": self._node_map.get(most_critical_node, str(most_critical_node)),
            "most_critical_edge":  most_critical_edge,
        }

    def all_paths_stats(self, source: int = 0, sink: int = 3) -> pd.DataFrame:
        """
        Enumerate all simple paths from source to sink with cost, transit time,
        total risk, and total distance.

        Returns
        -------
        pd.DataFrame — one row per path
        """
        try:
            all_paths = list(nx.all_simple_paths(self.G, source, sink))
        except nx.NetworkXError:
            return pd.DataFrame()

        rows = []
        for path in all_paths:
            cost = transit = risk = distance = 0.0
            for u, v in zip(path, path[1:]):
                attrs = self.G[u][v]
                cost     += attrs.get("cost_per_unit", 0)
                transit  += attrs.get("transit_time_mean_h", 0)
                risk     += attrs.get("transit_risk", 0)
                distance += attrs.get("distance_km", 0)
            rows.append({
                "path":           " → ".join(self._node_map.get(n, str(n)) for n in path),
                "total_cost":     round(cost, 2),
                "transit_time_h": round(transit, 2),
                "transit_risk":   round(risk / max(len(path) - 1, 1), 4),
                "distance_km":    round(distance, 2),
                "n_hops":         len(path) - 1,
            })
        return pd.DataFrame(rows).sort_values("total_cost")

    # ─────────────────────────────────────────────────────────────────────────
    # Exports
    # ─────────────────────────────────────────────────────────────────────────

    def export_adjacency_matrix(self, attribute: str = "cost") -> pd.DataFrame:
        """
        Export adjacency matrix as a Pandas DataFrame.

        Parameters
        ----------
        attribute : edge attribute to use as matrix values

        Returns
        -------
        pd.DataFrame — (n_nodes × n_nodes) matrix, 0 where no edge
        """
        node_ids = sorted(self.G.nodes())
        names    = [self._node_map.get(n, str(n)) for n in node_ids]
        mat      = pd.DataFrame(0.0, index=names, columns=names)

        for u, v, data in self.G.edges(data=True):
            u_name = self._node_map.get(u, str(u))
            v_name = self._node_map.get(v, str(v))
            mat.loc[u_name, v_name] = data.get(attribute, 0.0)

        return mat

    def to_dict(self) -> Dict:
        """Export the full network as a serialisable dict."""
        return {
            "nodes": [
                {"id": n, **dict(self.G.nodes[n])}
                for n in sorted(self.G.nodes())
            ],
            "edges": [
                {"from": u, "to": v, **dict(data)}
                for u, v, data in self.G.edges(data=True)
            ],
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Visualisation
    # ─────────────────────────────────────────────────────────────────────────

    def visualize_network(
        self,
        highlight_critical: bool = True,
        ax: Optional[plt.Axes] = None,
        title: str = "Cold Chain Network",
    ) -> plt.Figure:
        """
        Draw the cold chain network with:
          - Node size ∝ storage capacity
          - Node colour by type (source/hub/warehouse/sink)
          - Edge colour by transit risk (green→yellow→red)
          - Critical node highlighted with red border if highlight_critical=True
          - Edge labels showing transit time and distance

        Parameters
        ----------
        highlight_critical : draw red border around highest-betweenness node
        ax                 : optional existing axes
        title              : figure title

        Returns
        -------
        matplotlib Figure
        """
        fig, ax_ = (plt.subplots(figsize=(12, 5)) if ax is None else (ax.get_figure(), ax))
        ax_ = fig.axes[0] if ax is None else ax

        # Layout: use stored x,y if available, else spring layout
        pos = {}
        for nid in self.G.nodes():
            attrs = self.G.nodes[nid]
            if "x" in attrs and "y" in attrs:
                pos[nid] = (attrs["x"], attrs["y"])
        if not pos:
            pos = nx.spring_layout(self.G, seed=42)

        # Node sizes ∝ capacity (normalised)
        capacities = [self.G.nodes[n].get("capacity", 100) for n in self.G.nodes()]
        max_cap    = max(capacities)
        node_sizes = [800 + 2000 * (c / max_cap) for c in capacities]

        # Node colours by type
        node_colors = [
            NODE_COLORS.get(self.G.nodes[n].get("type", "hub"), "#aaaaaa")
            for n in self.G.nodes()
        ]

        # Edge colours by transit_risk (green=low, red=high)
        risks       = [self.G[u][v].get("transit_risk", 0.5) for u, v in self.G.edges()]
        edge_colors = [plt.cm.RdYlGn_r(r) for r in risks]
        edge_widths = [1.5 + 3.0 * r for r in risks]

        # Draw
        nx.draw_networkx_nodes(
            self.G, pos, ax=ax_,
            node_size=node_sizes, node_color=node_colors,
            edgecolors="black", linewidths=1.5, alpha=0.9,
        )
        nx.draw_networkx_labels(
            self.G, pos, ax=ax_,
            labels=self._node_map,
            font_size=7, font_weight="bold",
        )
        nx.draw_networkx_edges(
            self.G, pos, ax=ax_,
            edge_color=edge_colors, width=edge_widths,
            arrowsize=18, arrowstyle="-|>",
            connectionstyle="arc3,rad=0.05",
        )

        # Edge labels
        edge_labels = {
            (u, v): f"{self.G[u][v]['transit_time_mean_h']:.1f}h\n{self.G[u][v]['distance_km']:.0f}km"
            for u, v in self.G.edges()
        }
        nx.draw_networkx_edge_labels(
            self.G, pos, edge_labels=edge_labels, ax=ax_,
            font_size=6, alpha=0.8,
        )

        # Highlight critical node
        if highlight_critical:
            scores = self.criticality_scores()
            crit_node = scores["most_critical_node"]
            if crit_node in pos:
                ax_.scatter(
                    [pos[crit_node][0]], [pos[crit_node][1]],
                    s=node_sizes[list(self.G.nodes()).index(crit_node)] + 400,
                    color="red", alpha=0.25, zorder=0,
                )
                ax_.annotate(
                    "⚠ Critical", pos[crit_node],
                    xytext=(pos[crit_node][0], pos[crit_node][1] + 0.08),
                    fontsize=7, color="red", ha="center",
                )

        # Legend
        legend_elements = [
            mpatches.Patch(color=c, label=t.capitalize())
            for t, c in NODE_COLORS.items()
        ]
        legend_elements.append(
            mpatches.Patch(color=plt.cm.RdYlGn_r(0.8), label="High transit risk")
        )
        ax_.legend(handles=legend_elements, loc="lower right", fontsize=7)
        ax_.set_title(title, fontsize=11, fontweight="bold")
        ax_.axis("off")
        plt.tight_layout()
        return fig

    def __repr__(self) -> str:
        return (
            f"ColdChainNetwork(nodes={self.G.number_of_nodes()}, "
            f"edges={self.G.number_of_edges()})"
        )
