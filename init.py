"""
simulation package
==================
SimPy discrete-event simulation engine for cold chain logistics.

Processes:
  - demand_arrival     : stochastic demand at hospital node
  - replenishment_cycle: (Q,R) or (R,S) inventory review + ordering
  - temperature_monitoring: OU process + excursion detection
  - equipment_maintenance : failure scheduling + OU parameter switching
  - expiry_check       : daily FEFO expiry removal

Author: IIT Kharagpur, Dept. of Industrial and Systems Engineering
"""

from .cold_chain_sim import ColdChainSimulation
from .metrics import MetricsCollector
from .events import SimEvents

__all__ = ["ColdChainSimulation", "MetricsCollector", "SimEvents"]
