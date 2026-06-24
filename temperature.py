"""
models/temperature.py
=====================
Temperature risk models for cold chain simulation.

Three coupled models:
  1. OrnsteinUhlenbeckProcess  — stochastic temperature dynamics
  2. TemperatureExcursionDetector — zone-based excursion classification
  3. TimeTemperatureIndex (TTI)   — Arrhenius cumulative quality degradation
  4. EquipmentFailureModel        — Poisson failure / exponential repair

Theoretical backbone:
  - OU process: Ornstein & Uhlenbeck (1930) mean-reverting SDE
    dT = θ(μ - T)dt + σ dW
  - TTI / Arrhenius: Labuza (1984), "Application of Chemical Kinetics to
    Deterioration of Foods", J. Chemical Education 61(4), 348.
  - WHO PQS cold chain standards: WHO/PQS/E06/IN05.3

Key insight: OU process is chosen over a simple random walk because
refrigeration systems actively correct temperature deviations through
thermostatic control loops — mean reversion (parameter θ) captures this
control behaviour. A failed refrigerator has θ ≈ 0, allowing temperature
to drift toward ambient.

Author: IIT Kharagpur, Dept. of Industrial and Systems Engineering
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

DEFAULT_SEED = 42

# ─────────────────────────────────────────────────────────────────────────────
# Temperature zone thresholds (°C) — WHO blood storage guidelines
# ─────────────────────────────────────────────────────────────────────────────
ZONES = {
    "safe":     (2.0, 6.0),    # Target storage for blood products
    "warning":  [(1.0, 2.0), (6.0, 8.0)],
    "critical": [(-np.inf, 1.0), (8.0, np.inf)],
}

# Arrhenius constants for blood products (approximation)
BLOOD_ACTIVATION_ENERGY = 70_000  # J/mol (70 kJ/mol — typical for biological degradation)
GAS_CONSTANT = 8.314               # J/(mol·K)
REFERENCE_TEMP_K = 277.15          # 4°C in Kelvin (reference temperature)


# ─────────────────────────────────────────────────────────────────────────────
# 1. ORNSTEIN-UHLENBECK TEMPERATURE PROCESS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OUParameters:
    """Parameters for OU process under different operating conditions."""
    theta: float   # Mean reversion speed (h⁻¹ or min⁻¹)
    mu:    float   # Long-run mean temperature (°C)
    sigma: float   # Volatility (°C/√time)
    label: str     # Human-readable description

    # Preset parameter sets calibrated to cold chain equipment behaviour
    @staticmethod
    def normal() -> "OUParameters":
        """Normal refrigeration operation: strong mean reversion, low volatility."""
        return OUParameters(theta=2.0, mu=4.0, sigma=0.3, label="Normal Operation")

    @staticmethod
    def degraded() -> "OUParameters":
        """Degraded refrigeration (partial failure, aging compressor)."""
        return OUParameters(theta=0.5, mu=4.0, sigma=1.2, label="Degraded Refrigeration")

    @staticmethod
    def failed() -> "OUParameters":
        """
        Complete equipment failure: temperature drifts toward ambient (~22°C).
        theta ≈ 0 means almost no mean reversion (no active cooling).
        """
        return OUParameters(theta=0.05, mu=22.0, sigma=0.5, label="Equipment Failure")

    @staticmethod
    def transit(transit_quality: str = "good") -> "OUParameters":
        """Transit vehicle refrigeration (less precise than static storage)."""
        if transit_quality == "good":
            return OUParameters(theta=1.5, mu=4.0, sigma=0.6, label="Transit — Good")
        elif transit_quality == "poor":
            return OUParameters(theta=0.8, mu=5.0, sigma=1.5, label="Transit — Poor")
        else:
            return OUParameters(theta=0.3, mu=12.0, sigma=2.0, label="Transit — Broken AC")


class OrnsteinUhlenbeckProcess:
    """
    Ornstein-Uhlenbeck (mean-reverting) stochastic temperature process.

    The OU process models refrigerator temperature as:
        dT = θ(μ - T)dt + σ dW_t

    where:
        θ  = mean reversion speed — how strongly the refrigerator pulls
             temperature back toward the setpoint μ
        μ  = long-run mean (target temperature, e.g. 4°C for blood)
        σ  = diffusion coefficient (random perturbations from door openings,
             ambient heat leakage, compressor cycling)
        dW = Wiener process increment ~ N(0, dt)

    Exact discretisation (Euler-Maruyama scheme):
        T_{t+dt} = T_t × e^{-θdt} + μ(1 - e^{-θdt}) + σ √[(1-e^{-2θdt})/(2θ)] × ε_t

    Comment: "OU process is chosen over simple random walk because
    refrigeration systems actively correct temperature deviations —
    mean reversion captures this control behaviour. A thermostat is
    literally a mean-reverting controller: when T > μ, it increases cooling;
    when T < μ, it reduces cooling. The parameter θ quantifies control speed."

    Reference: Ornstein, L.S. & Uhlenbeck, G.E. (1930). "On the theory of
               Brownian motion." Physical Review, 36(5), 823-841.
    """

    def __init__(
        self,
        params: Optional[OUParameters] = None,
        seed: int = DEFAULT_SEED,
    ):
        """
        Parameters
        ----------
        params : OUParameters object (defaults to normal operation)
        seed   : random seed for reproducibility
        """
        self.params = params or OUParameters.normal()
        self.rng    = np.random.default_rng(seed)
        self._last_simulation: Optional[np.ndarray] = None
        self._last_time: Optional[np.ndarray] = None

    def simulate(
        self,
        T0: float,
        n_steps: int,
        dt: float = 1 / 60,  # 1 minute in hours
        params: Optional[OUParameters] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate OU temperature process using exact discretisation.

        Exact (non-Euler) discretisation of OU SDE:
            T_{t+Δt} = T_t × e^{-θΔt}  +  μ(1 - e^{-θΔt})
                       + σ × sqrt[(1 - e^{-2θΔt}) / (2θ)] × ε_t
            where ε_t ~ N(0, 1)   ... Gillespie (1996), exact OU simulation

        Parameters
        ----------
        T0      : initial temperature (°C)
        n_steps : number of time steps
        dt      : time step size (hours, default 1/60 = 1 minute)
        params  : override default parameters for this simulation

        Returns
        -------
        (time_array, temp_array) — both shape (n_steps,)
        """
        p = params or self.params
        theta, mu, sigma = p.theta, p.mu, p.sigma

        # Pre-compute OU exact-step constants
        exp_neg_theta_dt = np.exp(-theta * dt)

        # Variance of OU increment: σ²(1 - e^{-2θΔt}) / (2θ)
        # When θ → 0, this approaches σ²Δt (Brownian motion limit)
        if theta > 1e-9:
            var_increment = sigma**2 * (1 - np.exp(-2 * theta * dt)) / (2 * theta)
        else:
            var_increment = sigma**2 * dt
        std_increment = np.sqrt(max(0, var_increment))

        # Generate white noise increments
        noise = self.rng.standard_normal(n_steps)

        # Simulate step by step
        T = np.empty(n_steps)
        T[0] = T0
        for i in range(1, n_steps):
            # T_{t+dt} = T_t × e^{-θdt} + μ(1 - e^{-θdt}) + std_inc × ε
            T[i] = (T[i - 1] * exp_neg_theta_dt
                    + mu * (1 - exp_neg_theta_dt)
                    + std_increment * noise[i])

        time_arr = np.arange(n_steps) * dt
        self._last_simulation = T
        self._last_time       = time_arr
        return time_arr, T

    def simulate_with_failure(
        self,
        T0: float,
        total_hours: float,
        failure_start_h: float,
        repair_start_h: float,
        dt: float = 1 / 60,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate OU process with equipment failure episode.

        Three phases:
          [0, failure_start_h)    : normal OU parameters
          [failure_start_h, repair_start_h) : failed equipment parameters
          [repair_start_h, total_hours]     : normal OU parameters (restored)

        Returns (time, temp, phase_array) where phase ∈ {0, 1, 2}
        """
        n_total = int(total_hours / dt)
        n_fail  = int(failure_start_h / dt)
        n_repair = int(repair_start_h / dt)

        T_all    = np.empty(n_total)
        phase_all = np.zeros(n_total, dtype=int)

        # Phase 0: normal
        _, T_phase0 = self.simulate(T0, n_fail, dt, OUParameters.normal())
        T_all[:n_fail]   = T_phase0
        phase_all[:n_fail] = 0

        # Phase 1: failure
        _, T_phase1 = self.simulate(T_all[n_fail - 1], n_repair - n_fail, dt, OUParameters.failed())
        T_all[n_fail:n_repair]   = T_phase1
        phase_all[n_fail:n_repair] = 1

        # Phase 2: restored
        _, T_phase2 = self.simulate(T_all[n_repair - 1], n_total - n_repair, dt, OUParameters.normal())
        T_all[n_repair:]   = T_phase2
        phase_all[n_repair:] = 2

        time_arr = np.arange(n_total) * dt
        return time_arr, T_all, phase_all

    def stationary_distribution(self) -> Tuple[float, float]:
        """
        Return mean and std of the stationary (long-run) OU distribution.

        Stationary distribution: T∞ ~ N(μ, σ²/(2θ))
        This is the distribution temperature converges to in steady state.

        Returns (mean, std)
        """
        p    = self.params
        mean = p.mu
        std  = p.sigma / np.sqrt(2 * p.theta) if p.theta > 1e-9 else np.inf
        return mean, std

    def plot(
        self,
        time_arr: np.ndarray,
        temp_arr: np.ndarray,
        ax: Optional[plt.Axes] = None,
        show_zones: bool = True,
    ) -> plt.Figure:
        """Plot temperature time series with zone bands."""
        fig, ax_ = (plt.subplots(figsize=(12, 4)) if ax is None else (ax.get_figure(), ax))
        ax_ = fig.axes[0] if ax is None else ax

        ax_.plot(time_arr, temp_arr, lw=0.8, color="#2c7bb6", alpha=0.85,
                 label=f"Temperature — {self.params.label}")
        ax_.axhline(self.params.mu, color="#2c7bb6", lw=1.0, linestyle=":",
                    alpha=0.5, label=f"Target μ={self.params.mu}°C")

        if show_zones:
            # Safe zone (green band)
            ax_.axhspan(2.0, 6.0, alpha=0.12, color="green", label="Safe (2–6°C)")
            # Warning zones (yellow)
            ax_.axhspan(1.0, 2.0, alpha=0.15, color="orange", label="Warning")
            ax_.axhspan(6.0, 8.0, alpha=0.15, color="orange")
            # Critical zones (red)
            lo = ax_.get_ylim()[0] if ax_.get_ylim()[0] < 1.0 else -5
            ax_.axhspan(lo, 1.0, alpha=0.15, color="red", label="Critical")
            ax_.axhspan(8.0, max(temp_arr.max() + 1, 9), alpha=0.15, color="red")

        ax_.set_xlabel("Time (hours)")
        ax_.set_ylabel("Temperature (°C)")
        ax_.set_title(f"OU Temperature Process: dT = θ(μ−T)dt + σdW\n"
                      f"θ={self.params.theta}, μ={self.params.mu}, σ={self.params.sigma}")
        ax_.legend(fontsize=8, loc="upper right")
        ax_.grid(alpha=0.3)
        plt.tight_layout()
        return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. TEMPERATURE EXCURSION DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExcursionEvent:
    """Records a single temperature excursion event."""
    start_step:      int
    end_step:        int
    start_time_h:    float
    end_time_h:      float
    max_temp:        float
    min_temp:        float
    zone:            str    # 'warning' or 'critical'
    duration_minutes: float


class TemperatureExcursionDetector:
    """
    Detects and classifies temperature excursion events from OU process output.

    Zone definitions (WHO blood storage guidelines):
        Safe zone:     2°C ≤ T ≤ 6°C    → no action required
        Warning zone:  1°C ≤ T < 2°C
                    OR 6°C < T ≤ 8°C    → monitor closely, assess product
        Critical zone: T < 1°C
                    OR T > 8°C           → reject batch, trigger contingency

    Excursion detection logic:
        - An excursion begins when T first leaves the safe zone
        - It ends when T returns to the safe zone
        - Overlapping warning→critical within one excursion episode is
          classified at the highest severity reached

    Reference: WHO (2019). "Manual on the Management, Maintenance and Use
               of Blood Cold Chain Equipment." WHO/EMP/2019.02
    """

    def __init__(self):
        self.safe_lo  = 2.0
        self.safe_hi  = 6.0
        self.warn_lo  = 1.0
        self.warn_hi  = 8.0

    def classify_temperature(self, T: float) -> str:
        """
        Classify a single temperature reading into a zone.

        Returns
        -------
        str: 'safe', 'warning', or 'critical'
        """
        if self.safe_lo <= T <= self.safe_hi:
            return "safe"
        elif self.warn_lo <= T <= self.warn_hi:
            return "warning"
        else:
            return "critical"

    def detect_excursions(
        self,
        time_arr: np.ndarray,
        temp_series: np.ndarray,
        dt_hours: float = 1 / 60,
    ) -> List[ExcursionEvent]:
        """
        Scan temperature series and return list of excursion events.

        Parameters
        ----------
        time_arr   : time axis (hours)
        temp_series: temperature readings (°C), same length as time_arr
        dt_hours   : time step in hours (used for duration calculation)

        Returns
        -------
        List of ExcursionEvent objects, one per distinct excursion episode.
        """
        zones = np.array([self.classify_temperature(t) for t in temp_series])
        excursions: List[ExcursionEvent] = []

        in_excursion  = False
        exc_start     = 0
        exc_max_temp  = -np.inf
        exc_min_temp  =  np.inf
        exc_zone      = "warning"

        for i, (z, T) in enumerate(zip(zones, temp_series)):
            if z != "safe":
                if not in_excursion:
                    in_excursion = True
                    exc_start    = i
                    exc_max_temp = T
                    exc_min_temp = T
                    exc_zone     = z
                else:
                    exc_max_temp = max(exc_max_temp, T)
                    exc_min_temp = min(exc_min_temp, T)
                    # Escalate zone if critical reached during episode
                    if z == "critical":
                        exc_zone = "critical"
            else:
                if in_excursion:
                    duration_min = (i - exc_start) * dt_hours * 60
                    excursions.append(ExcursionEvent(
                        start_step=exc_start,
                        end_step=i,
                        start_time_h=time_arr[exc_start],
                        end_time_h=time_arr[i],
                        max_temp=round(exc_max_temp, 3),
                        min_temp=round(exc_min_temp, 3),
                        zone=exc_zone,
                        duration_minutes=round(duration_min, 1),
                    ))
                    in_excursion = False

        # Close any open excursion at end of series
        if in_excursion:
            duration_min = (len(temp_series) - exc_start) * dt_hours * 60
            excursions.append(ExcursionEvent(
                start_step=exc_start,
                end_step=len(temp_series) - 1,
                start_time_h=time_arr[exc_start],
                end_time_h=time_arr[-1],
                max_temp=round(exc_max_temp, 3),
                min_temp=round(exc_min_temp, 3),
                zone=exc_zone,
                duration_minutes=round(duration_min, 1),
            ))

        return excursions

    def excursion_probability(
        self,
        ou_params: OUParameters,
        transit_duration_h: float = 4.0,
        dt: float = 1 / 60,
        T0: float = 4.0,
        n_simulations: int = 1000,
        seed: int = DEFAULT_SEED,
    ) -> Dict:
        """
        Monte Carlo estimate of P(at least one excursion) during a transit.

        Runs n_simulations OU trajectories over transit_duration_h and checks
        whether each trajectory experiences at least one zone excursion.

        Parameters
        ----------
        ou_params          : OU process parameters for this transit
        transit_duration_h : hours of transit
        dt                 : simulation time step (hours)
        T0                 : initial temperature (°C)
        n_simulations      : Monte Carlo replications

        Returns
        -------
        dict: {p_warning, p_critical, p_any_excursion, mean_excursion_duration_min,
               n_critical_excursions}
        """
        n_steps = int(transit_duration_h / dt)
        rng     = np.random.default_rng(seed)
        proc    = OrnsteinUhlenbeckProcess(ou_params, seed=seed)

        n_warning   = 0
        n_critical  = 0
        durations   = []

        for sim in range(n_simulations):
            # Use different seed for each simulation run
            proc.rng = np.random.default_rng(seed + sim)
            _, T_sim = proc.simulate(T0, n_steps, dt)
            excursions = self.detect_excursions(np.arange(n_steps) * dt, T_sim, dt)

            if any(e.zone == "warning" or e.zone == "critical" for e in excursions):
                n_warning += 1
            if any(e.zone == "critical" for e in excursions):
                n_critical += 1
                durations.extend([e.duration_minutes for e in excursions if e.zone == "critical"])

        return {
            "p_any_excursion":           round(n_warning / n_simulations, 4),
            "p_critical_excursion":      round(n_critical / n_simulations, 4),
            "p_warning_only":            round((n_warning - n_critical) / n_simulations, 4),
            "mean_critical_duration_min": round(np.mean(durations), 2) if durations else 0.0,
            "transit_duration_h":        transit_duration_h,
            "n_simulations":             n_simulations,
        }

    def excursion_summary_df(self, excursions: List[ExcursionEvent]) -> pd.DataFrame:
        """Convert list of ExcursionEvent objects to a tidy DataFrame."""
        if not excursions:
            return pd.DataFrame(columns=[
                "start_time_h", "end_time_h", "duration_minutes",
                "max_temp", "min_temp", "zone"
            ])
        return pd.DataFrame([{
            "start_time_h":     e.start_time_h,
            "end_time_h":       e.end_time_h,
            "duration_minutes": e.duration_minutes,
            "max_temp":         e.max_temp,
            "min_temp":         e.min_temp,
            "zone":             e.zone,
        } for e in excursions])


# ─────────────────────────────────────────────────────────────────────────────
# 3. TIME-TEMPERATURE INDEX (TTI) MODEL
# ─────────────────────────────────────────────────────────────────────────────

class TimeTemperatureIndex:
    """
    Cumulative product quality degradation via Arrhenius kinetic model (TTI).

    Biological and chemical degradation rates follow Arrhenius kinetics:
        k(T) = A × exp(−Ea / (R × T_K))
    where:
        k(T) = degradation rate constant at temperature T (in Kelvin)
        A    = pre-exponential (Arrhenius) frequency factor
        Ea   = activation energy (J/mol) — barrier to degradation
        R    = 8.314 J/(mol·K) — universal gas constant
        T_K  = absolute temperature (°C + 273.15)

    Quality decay model:
        dQ/dt = −k(T) × Q   (first-order kinetics)
        Q(t)  = Q0 × exp(−∫ k(T(τ)) dτ)

    Equivalently, the cumulative degradation integral D:
        D = ∫₀ᵗ k(T(τ)) dτ
        Quality index Q = exp(−D)   (starts at 1.0, decays toward 0)

    Comment: "TTI model is used in pharmaceutical cold chain monitoring
    (WHO PQS standards) to assess cumulative thermal damage to products.
    A product may experience brief high-temperature excursions and still
    be usable if the cumulative degradation remains below threshold."

    Reference: Labuza, T.P. (1984). "Application of Chemical Kinetics to
               Deterioration of Foods." J. Chemical Education, 61(4), 348-358.
               WHO/PQS/E06/IN05.3 — Temperature monitoring device standards.
    """

    def __init__(
        self,
        activation_energy: float = BLOOD_ACTIVATION_ENERGY,
        pre_exponential: Optional[float] = None,
        reference_temp_c: float = 4.0,
        reference_rate:   float = 1.0 / (42 * 24),  # 1/lifetime in hours for packed RBC
    ):
        """
        Parameters
        ----------
        activation_energy : Ea in J/mol (default 70,000 J/mol for blood products)
        pre_exponential   : A factor; if None, calibrated to reference_rate at reference_temp_c
        reference_temp_c  : temperature (°C) at which reference_rate applies
        reference_rate    : degradation rate (1/hour) at reference temperature
        """
        self.Ea  = activation_energy
        self.R   = GAS_CONSTANT
        self.T0  = reference_temp_c + 273.15  # reference temperature in Kelvin

        # Calibrate A from reference rate: k(T0) = A × exp(−Ea/RT0) = reference_rate
        # → A = reference_rate / exp(−Ea/RT0) = reference_rate × exp(Ea/RT0)
        if pre_exponential is None:
            self.A = reference_rate * np.exp(self.Ea / (self.R * self.T0))
        else:
            self.A = pre_exponential

    def rate_constant(self, temp_c: float) -> float:
        """
        Arrhenius degradation rate constant at temperature temp_c (°C).

            k(T) = A × exp(−Ea / (R × T_K))   ... Arrhenius equation

        Parameters
        ----------
        temp_c : temperature in Celsius

        Returns
        -------
        float — degradation rate constant (per hour)
        """
        T_K = temp_c + 273.15
        # k(T) = A × exp(−Ea/RT) — Arrhenius rate constant
        return self.A * np.exp(-self.Ea / (self.R * T_K))

    def quality_decay(
        self,
        temp_series: np.ndarray,
        dt_minutes: float = 1.0,
    ) -> np.ndarray:
        """
        Compute quality index time series from temperature history.

            D(t) = Σ k(T_i) × dt    (cumulative degradation integral)
            Q(t) = exp(−D(t))        (quality index ∈ [0, 1])

        Parameters
        ----------
        temp_series : temperature readings (°C), shape (n,)
        dt_minutes  : time step in minutes (default 1 minute)

        Returns
        -------
        quality_series : shape (n,), values in [0, 1]
                         1.0 = perfect quality, 0 = fully degraded
        """
        dt_hours = dt_minutes / 60.0
        # Degradation rate at each step
        k_series = np.array([self.rate_constant(T) for T in temp_series])
        # Cumulative degradation integral: D(t) = cumsum(k × dt)
        D_cumulative = np.cumsum(k_series * dt_hours)
        # Quality index: Q(t) = exp(−D(t))
        quality = np.exp(-D_cumulative)
        return quality

    def is_usable(self, quality_index: float, threshold: float = 0.8) -> bool:
        """
        Determine if product is still usable based on quality index.

        Parameters
        ----------
        quality_index : float ∈ [0, 1]
        threshold     : minimum acceptable quality (default 0.8 — 20% degradation)

        Returns
        -------
        bool — True if quality ≥ threshold
        """
        return quality_index >= threshold

    def residual_shelf_life(
        self,
        temp_history: np.ndarray,
        base_shelf_life_hours: float,
        dt_minutes: float = 1.0,
        quality_threshold: float = 0.8,
    ) -> float:
        """
        Estimate remaining shelf life after observed temperature history.

        Method:
          1. Compute cumulative degradation D_so_far from temp_history
          2. At reference temperature T_ref, find time t_remaining such that
             D_so_far + k(T_ref) × t_remaining = D_max_allowed
          3. D_max_allowed = −ln(quality_threshold)

        Parameters
        ----------
        temp_history          : observed temperature series (°C)
        base_shelf_life_hours : nominal shelf life at reference temp (hours)
        dt_minutes            : time step of temp_history
        quality_threshold     : minimum acceptable quality

        Returns
        -------
        float — estimated residual shelf life in hours
        """
        dt_hours  = dt_minutes / 60.0
        k_series  = np.array([self.rate_constant(T) for T in temp_history])
        D_so_far  = np.sum(k_series * dt_hours)

        # D_max = degradation at expiry under reference conditions
        # = k(T_ref) × base_shelf_life_hours
        k_ref     = self.rate_constant(4.0)   # reference: 4°C
        D_max     = k_ref * base_shelf_life_hours

        D_allowed = D_max - D_so_far
        if D_allowed <= 0:
            return 0.0   # product already beyond useful life

        # Remaining time at reference temperature
        t_remaining_hours = D_allowed / k_ref
        return max(0.0, round(t_remaining_hours, 2))

    def q10_factor(self, delta_T: float = 10.0) -> float:
        """
        Q10 factor: how much faster degradation proceeds per 10°C rise.

            Q10 = k(T + 10) / k(T) = exp(Ea × ΔT / (R × T² × (T+ΔT)))

        Simplified (constant Ea): Q10 ≈ exp(Ea × ΔT / (R × T_avg²))
        For blood products: Q10 ≈ 2-3 (degradation doubles every 10°C)

        Returns
        -------
        float — Q10 factor
        """
        T_ref_K = REFERENCE_TEMP_K
        T_high_K = T_ref_K + delta_T
        # Q10 = k(T+10) / k(T) = exp(Ea/R × (1/T - 1/(T+ΔT)))
        q10 = np.exp(self.Ea / self.R * (1.0 / T_ref_K - 1.0 / T_high_K))
        return round(q10, 3)

    def plot_quality_decay(
        self,
        time_arr: np.ndarray,
        temp_arr: np.ndarray,
        dt_minutes: float = 1.0,
        ax: Optional[plt.Axes] = None,
        product_name: str = "Blood Product",
    ) -> plt.Figure:
        """
        Dual-axis plot: temperature (top) and quality decay (bottom).
        """
        quality = self.quality_decay(temp_arr, dt_minutes)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

        # Temperature
        ax1.plot(time_arr, temp_arr, color="#2c7bb6", lw=0.8)
        ax1.axhspan(2.0, 6.0, alpha=0.12, color="green")
        ax1.axhspan(1.0, 2.0, alpha=0.15, color="orange")
        ax1.axhspan(6.0, 8.0, alpha=0.15, color="orange")
        ax1.set_ylabel("Temperature (°C)")
        ax1.grid(alpha=0.3)
        ax1.set_title(f"TTI Quality Decay — {product_name}\n"
                      "k(T) = A·exp(−Ea/RT) [Arrhenius]; Q(t) = exp(−∫k(T)dt)")

        # Quality
        ax2.plot(time_arr, quality, color="#d7191c", lw=1.5, label="Quality Index Q(t)")
        ax2.axhline(0.8, color="orange", lw=1.2, linestyle="--", label="Usability threshold (0.8)")
        ax2.axhline(1.0, color="green",  lw=0.8, linestyle=":", alpha=0.5)
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_ylabel("Quality Index Q(t)")
        ax2.set_xlabel("Time (hours)")
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. EQUIPMENT FAILURE MODEL
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FailureEvent:
    """Records a single equipment failure and repair event."""
    failure_time_h:    float   # when failure occurs (simulation hours)
    repair_start_h:    float   # when repair begins (same as failure for simplicity)
    repair_end_h:      float   # when equipment is restored
    repair_duration_h: float   # repair time
    node_id:           str


class EquipmentFailureModel:
    """
    Refrigeration equipment failure model based on Poisson process + exponential repair.

    Assumptions:
        - Failures arrive as a Poisson process with rate λ = 1/MTTF
          (MTTF = Mean Time To Failure)
        - Repair times are Exponential(μ_repair) — standard M/G/1 machinery model
        - Between failures, equipment operates normally (OU normal parameters)
        - During failure, OU parameters switch to 'failed' preset

    Poisson failures: P(k failures in t hours) = (λt)^k × e^{−λt} / k!
    Inter-arrival times: Exp(λ) — memoryless property of Poisson process

    Reference: Rausand, M. & Hoyland, A. (2004). "System Reliability Theory:
               Models, Statistical Methods and Applications." 2nd ed. Wiley.
    """

    def __init__(
        self,
        mttf_days: float = 180.0,       # Mean Time To Failure (days)
        mean_repair_hours: float = 4.0, # Mean repair duration (hours)
        node_id: str = "node",
        seed: int = DEFAULT_SEED,
    ):
        """
        Parameters
        ----------
        mttf_days         : MTTF in days (e.g. 180 = fails ~twice a year on average)
        mean_repair_hours : mean repair duration (hours); Exponential distribution
        node_id           : identifier for logging
        seed              : random seed
        """
        self.mttf_days         = mttf_days
        self.mttf_hours        = mttf_days * 24.0
        self.mean_repair_hours = mean_repair_hours
        self.node_id           = node_id
        self.rng               = np.random.default_rng(seed)
        self._failure_schedule: List[FailureEvent] = []

    def failure_times(self, n_years: float = 2.0) -> List[FailureEvent]:
        """
        Generate equipment failure schedule over n_years.

        Failure inter-arrival times: X_i ~ Exp(λ), λ = 1/MTTF_hours
        Repair durations: R_i ~ Exp(1/mean_repair_hours)

        Parameters
        ----------
        n_years : simulation horizon (years)

        Returns
        -------
        List of FailureEvent objects covering n_years
        """
        total_hours = n_years * 365 * 24
        # λ = 1/MTTF — Poisson process failure rate
        lam = 1.0 / self.mttf_hours

        events: List[FailureEvent] = []
        t = 0.0

        while t < total_hours:
            # Inter-arrival time: Exp(λ)
            inter_arrival = self.rng.exponential(scale=self.mttf_hours)
            t_fail = t + inter_arrival
            if t_fail >= total_hours:
                break

            # Repair duration: Exp(μ_repair)
            repair_dur = self.rng.exponential(scale=self.mean_repair_hours)
            t_repaired = t_fail + repair_dur

            events.append(FailureEvent(
                failure_time_h=round(t_fail, 2),
                repair_start_h=round(t_fail, 2),
                repair_end_h=round(t_repaired, 2),
                repair_duration_h=round(repair_dur, 2),
                node_id=self.node_id,
            ))
            t = t_repaired  # next failure can only happen after repair

        self._failure_schedule = events
        return events

    def is_operational(self, t_hours: float) -> bool:
        """
        Check whether equipment is operational at time t_hours.

        Parameters
        ----------
        t_hours : current simulation time in hours

        Returns
        -------
        bool — True if no active failure at time t
        """
        for event in self._failure_schedule:
            if event.failure_time_h <= t_hours < event.repair_end_h:
                return False
        return True

    def get_ou_params(self, t_hours: float) -> OUParameters:
        """
        Return OU parameters appropriate for equipment state at time t_hours.

        Normal operation → OUParameters.normal()
        During failure   → OUParameters.failed()
        After repair     → OUParameters.normal() restored
        """
        return OUParameters.normal() if self.is_operational(t_hours) else OUParameters.failed()

    def repair_duration(self) -> float:
        """
        Sample a single repair duration from Exp(mean_repair_hours).

        Returns
        -------
        float — repair time in hours
        """
        return float(self.rng.exponential(scale=self.mean_repair_hours))

    def availability(self, n_years: float = 2.0) -> float:
        """
        Estimate equipment availability A = MTTF / (MTTF + MTTR).

        This is the theoretical availability from renewal theory.

        Returns
        -------
        float ∈ [0, 1] — fraction of time equipment is operational
        """
        mttr = self.mean_repair_hours
        return self.mttf_hours / (self.mttf_hours + mttr)

    def summary(self, n_years: float = 2.0) -> Dict:
        """
        Summary statistics for equipment reliability over n_years.

        Returns
        -------
        dict: {n_failures, total_downtime_h, availability, expected_failures}
        """
        events = self.failure_times(n_years)
        total_downtime = sum(e.repair_duration_h for e in events)
        total_hours    = n_years * 365 * 24
        expected_failures = total_hours / self.mttf_hours

        return {
            "n_failures_simulated": len(events),
            "expected_failures":    round(expected_failures, 1),
            "total_downtime_h":     round(total_downtime, 1),
            "mean_repair_h":        round(np.mean([e.repair_duration_h for e in events]), 2) if events else 0,
            "availability":         round(self.availability(), 4),
            "mttf_days":            self.mttf_days,
        }
