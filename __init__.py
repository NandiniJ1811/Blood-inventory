"""
Cold Chain Inventory & Temperature Risk Simulation System
=========================================================
models package: inventory theory + temperature + network + perishability + contingency

Academic backbone: Newsvendor (Nahmias 1975), (Q,R) policy (Silver et al. 1998),
Ornstein-Uhlenbeck SDE, Arrhenius TTI model, Bellman DP.
"""

from .inventory import NewsvendorModel, EOQModel, ContinuousReviewQR, FEFOPolicy
from .temperature import OrnsteinUhlenbeckProcess, TemperatureExcursionDetector, TimeTemperatureIndex, EquipmentFailureModel
from .network import ColdChainNetwork
from .perishability import PerishableProduct
from .contingency import ContingencyProtocol

__all__ = [
    "NewsvendorModel",
    "EOQModel",
    "ContinuousReviewQR",
    "FEFOPolicy",
    "OrnsteinUhlenbeckProcess",
    "TemperatureExcursionDetector",
    "TimeTemperatureIndex",
    "EquipmentFailureModel",
    "ColdChainNetwork",
    "PerishableProduct",
    "ContingencyProtocol",
]
