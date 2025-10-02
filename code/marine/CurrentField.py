# ===============================================================================
# CurrentField Class
#
# Author: José Antonio González Prieto
# Date: 01/11/2025
# Version: 1.0
# Description:
#   This class provides multiple ocean current models (e.g., uniform,
#   sinusoidal, vortex) for use in Zermelo navigation simulations. Each
#   current model can be initialized stochastically using a dedicated
#   Numpy random number generator to ensure reproducibility. The class is
#   configurable via an external JSON file and supports both NumPy and Pyomo
#   backends for numerical and symbolic computations.
# ===============================================================================
from __future__ import annotations

# =======================================================================
# IMPORTS
# =======================================================================
import json
from typing import Optional, List, Tuple, Dict, Any, Callable
import numpy as np
import pyomo.environ as pyo
from utils.Geometry import Geometry


class CurrentField(Geometry):
    """Generates and manages various current fields for simulations."""

    SAFE_EPSILON: float = 1e-6

    def __init__(
        self,
        scenario_size: Tuple[float, float] = (200.0, 200.0),
        current_type: str = "uniform",
        velocity: float = 1.0,
        rng: Optional[np.random.Generator] = None,
        library_type: Optional[str] = None,
        time_varying: bool = False,
        config_file: str = "currents.json",
    ) -> None:
        """Initialize a CurrentField instance.

        Parameters
        ----------
        scenario_size : Tuple[float, float], optional
            Dimensions of the scenario (width, height), by default (200.0, 200.0).
        current_type : str, optional
            Type of current field to generate, by default "uniform".
        velocity : float, optional
            Base current velocity magnitude, by default 1.0.
        rng : Optional[np.random.Generator], optional
            Random number generator for deterministic initialization. If None, a
            new one is created, by default None.
        library_type : Optional[str], optional
            Backend for symbolic libraries (e.g., 'pyo'), by default None.
        time_varying : bool, optional
            If True, enables time-dependent behavior, by default False.
        config_file : str, optional
            Path to the JSON file with current parameters, by default "currents.json".

        Raises
        ------
        Exception
            Propagates any error that occurs during initialization.
        """
        try:
            super().__init__()
            self.rng: np.random.Generator = rng if rng is not None else np.random.default_rng()
            self.config: Dict[str, Any] = self._loadConfig(config_file)
            self._time_varying: bool = time_varying
            self.reset(library_type=library_type)
            self.scenario_size_x: float = scenario_size[0]
            self.scenario_size_y: float = scenario_size[1]
            self.velocity: float = velocity
            self.setCurrentType(current_type)
        except Exception as e:
            print(f"[ERROR] CurrentField.__init__ failed: {e}")
            raise

    def _loadConfig(self, config_file: str) -> Dict[str, Any]:
        """Load the configuration file for current parameters.

        Parameters
        ----------
        config_file : str
            The path to the JSON configuration file.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the loaded configuration.

        Raises
        ------
        Exception
            Propagates file not found or JSON decoding errors.
        """
        try:
            with open(config_file, 'r') as f:
                config_data: Dict[str, Any] = json.load(f)
                return config_data
        except Exception as e:
            print(f"[ERROR] CurrentField._loadConfig failed: {e}")
            raise

    def setTimeVarying(self, enabled: bool) -> None:
        """Enable or disable time-varying behavior for the current field.

        Parameters
        ----------
        enabled : bool
            Set to True to enable time-dependency, False to disable.
        
        Raises
        ------
        Exception
            Propagates any underlying error.
        """
        try:
            self._time_varying = enabled
        except Exception as e:
            print(f"[ERROR] CurrentField.setTimeVarying failed: {e}")
            raise

    def getTimeVarying(self) -> bool:
        """Check if time-varying behavior is enabled.

        Returns
        -------
        bool
            True if time-dependency is enabled, otherwise False.

        Raises
        ------
        Exception
            Propagates any underlying error.
        """
        try:
            return self._time_varying
        except Exception as e:
            print(f"[ERROR] CurrentField.getTimeVarying failed: {e}")
            raise

    def _vec(self, u: Any, v: Any) -> np.ndarray:
        """Build a NumPy vector [u, v] with a suitable dtype.

        Parameters
        ----------
        u : Any
            The u-component (horizontal) of the vector.
        v : Any
            The v-component (vertical) of the vector.

        Returns
        -------
        np.ndarray
            The resulting vector as a NumPy array.

        Raises
        ------
        Exception
            Propagates any underlying error.
        """
        try:
            if self.library_type is None:
                return np.array([float(u), float(v)], dtype=float)
            return np.array([u, v], dtype=object)
        except Exception as e:
            print(f"[ERROR] CurrentField._vec failed: {e}")
            raise

    def getVelocity(self) -> float:
        """Get the base velocity magnitude of the field.

        Returns
        -------
        float
            The current base velocity.

        Raises
        ------
        Exception
            Propagates any underlying error.
        """
        try:
            return self.velocity
        except Exception as e:
            print(f"[ERROR] CurrentField.getVelocity failed: {e}")
            raise

    def setVelocity(self, velocity: float) -> None:
        """Set the base velocity magnitude of the field.

        Parameters
        ----------
        velocity : float
            The new base velocity to set.

        Raises
        ------
        Exception
            Propagates any underlying error.
        """
        try:
            self.velocity = velocity
        except Exception as e:
            print(f"[ERROR] CurrentField.setVelocity failed: {e}")
            raise

    @staticmethod
    def getCurrentTypes() -> List[str]:
        """Get the list of available current types.

        Returns
        -------
        List[str]
            A list of strings with the names of supported current types.
        """
        return [
            "uniform", "sinusoidal", "logarithmic", "gaussianSwir", "vortex",
            "karmanVortex", "coastalTidal", "linearShear", "doubleGyre",
            "gaussianJet", "riverOutflow", "turbulenceNoise",
        ]

    def setCurrentType(self, current_type: str) -> None:
        """Assign the current type by name and reset its parameters.

        Parameters
        ----------
        current_type : str
            The name of the current type to set.

        Raises
        ------
        ValueError
            If the current type is not found in the configuration or is unknown.
        Exception
            Propagates any other underlying error.
        """
        try:
            type_mapping: Dict[str, int] = {
                "uniform": 0, "sinusoidal": 1, "vortex": 2, "logarithmic": 3,
                "gaussianSwir": 4, "karmanVortex": 5, "coastalTidal": 6,
                "linearShear": 7, "doubleGyre": 8, "gaussianJet": 9,
                "riverOutflow": 10, "turbulenceNoise": 11,
            }
            if current_type not in self.config:
                raise ValueError(f"Config for current type '{current_type}' not in JSON.")
            
            self.current_type: str = current_type
            self.current_type_id: int = type_mapping.get(current_type, -1)
            if self.current_type_id == -1:
                raise ValueError(f"Unknown current type: {current_type}")
            
            self.reset(self.library_type)
        except Exception as e:
            print(f"[ERROR] CurrentField.setCurrentType failed: {e}")
            raise

    def getCurrentTypeId(self) -> int:
        """Get the integer identifier of the current type.

        Returns
        -------
        int
            The integer ID of the current type.

        Raises
        ------
        Exception
            Propagates any underlying error.
        """
        try:
            return self.current_type_id
        except Exception as e:
            print(f"[ERROR] CurrentField.getCurrentTypeId failed: {e}")
            raise

    def getCurrentType(self) -> str:
        """Get the name of the current type.

        Returns
        -------
        str
            The name of the current type.
            
        Raises
        ------
        Exception
            Propagates any underlying error.
        """
        try:
            return self.current_type
        except Exception as e:
            print(f"[ERROR] CurrentField.getCurrentType failed: {e}")
            raise

    def setLibraryType(self, library_type: Optional[str]) -> None:
        """Set the backend library type (e.g., 'pyo' for Pyomo).

        Parameters
        ----------
        library_type : Optional[str]
            The name of the library backend.

        Raises
        ------
        Exception
            Propagates any underlying error.
        """
        try:
            self.library_type = library_type
        except Exception as e:
            print(f"[ERROR] CurrentField.setLibraryType failed: {e}")
            raise

    def reset(self, library_type: Optional[str] = None) -> None:
        """Reset all randomized parameters of the current field.

        Parameters
        ----------
        library_type : Optional[str], optional
            The library backend to use, by default None.

        Raises
        ------
        Exception
            Propagates any underlying error.
        """
        try:
            self.library_type = library_type
            self.uniform_random_angle: Optional[float] = None
            self.uniform_time_factor: Optional[float] = None
            self.sinusoidal_random_periods: Optional[np.ndarray] = None
            self.sinusoidal_random_gains: Optional[np.ndarray] = None
            self.sinusoidal_random_phases: Optional[np.ndarray] = None
            self.sinusoidal_time_freqs: Optional[np.ndarray] = None
            self.vortex_random_centers: Optional[np.ndarray] = None
            self.vortex_random_gains: Optional[np.ndarray] = None
            self.vortex_random_number: Optional[int] = None
            self.vortex_time_velocities: Optional[np.ndarray] = None
            self.logarithmic_random_number: Optional[int] = None
            self.logarithmic_random_centers: Optional[np.ndarray] = None
            self.logarithmic_random_rotations: Optional[np.ndarray] = None
            self.gaussian_swirl_random_centers: Optional[np.ndarray] = None
            self.gaussian_swirl_random_scales: Optional[np.ndarray] = None
            self.gaussian_swirl_random_gains: Optional[np.ndarray] = None
            self.gaussian_swirl_random_number: Optional[int] = None
            self.gaussian_swirl_time_velocities: Optional[np.ndarray] = None
            self.karman_spacing: Optional[float] = None
            self.karman_strength: Optional[float] = None
            self.karman_phase: Optional[float] = None
            self.karman_wake_center: Optional[float] = None
            self.karman_time_freq: Optional[float] = None
            self.tidal_period: Optional[float] = None
            self.tidal_phase: Optional[float] = None
            self.tidal_coast_y0: Optional[float] = None
            self.tidal_decay_scale: Optional[float] = None
            self.shear_a: Optional[float] = None
            self.shear_b: Optional[float] = None
            self.shear_cross: Optional[float] = None
            self.double_gyre_A: Optional[float] = None
            self.double_gyre_eps: Optional[float] = None
            self.double_gyre_time_freq: Optional[float] = None
            self.jet_sigma: Optional[float] = None
            self.jet_center_y: Optional[float] = None
            self.jet_axis: Optional[str] = None
            self.river_mouth: Optional[np.ndarray] = None
            self.river_spread: Optional[float] = None
            self.river_decay: Optional[float] = None
            self.turb_k: Optional[np.ndarray] = None
            self.turb_phases: Optional[np.ndarray] = None
            self.turb_gains: Optional[np.ndarray] = None
            self.turb_time_freqs: Optional[np.ndarray] = None
        except Exception as e:
            print(f"[ERROR] CurrentField.reset failed: {e}")
            raise

    def getCurrentAtPosition(self, x: float, y: float, t: float = 0.0) -> np.ndarray:
        """Compute the current vector at a specific position and time.

        Parameters
        ----------
        x : float
            The horizontal coordinate.
        y : float
            The vertical coordinate.
        t : float, optional
            The time coordinate, by default 0.0.

        Returns
        -------
        np.ndarray
            The current vector [u, v].

        Raises
        ------
        Exception
            Propagates any underlying error.
        """
        try:
            current_functions: Dict[int, Callable] = {
                0: self._uniformRandomCurrent, 1: self._sinusoidalRandomCurrent,
                2: self._vortexRandomCurrent, 3: self._logarithmicRandomFlowCurrent,
                4: self._gaussianSwirlRandomCurrent, 5: self._karmanVortexStreetCurrent,
                6: self._coastalTidalCurrent, 7: self._linearShearCurrent,
                8: self._doubleGyreCurrent, 9: self._gaussianJetCurrent,
                10: self._riverOutflowCurrent, 11: self._turbulenceNoiseCurrent,
            }
            func: Optional[Callable] = current_functions.get(self.current_type_id)
            if func is None:
                raise RuntimeError(f"Invalid current_type_id: {self.current_type_id}")
            return func(x, y, t)
        except Exception as e:
            print(f"[ERROR] CurrentField.getCurrentAtPosition failed: {e}")
            raise

    # ------------------------------------------------------------------
    # Private Current Field Implementations
    # ------------------------------------------------------------------

    def _uniformRandomCurrent(self, x: float, y: float, t: float) -> np.ndarray:
        """Compute a uniform current with a random angle."""
        try:
            if self.uniform_random_angle is None:
                params: Dict = self.config['uniform']
                self.uniform_random_angle = float(self.rng.uniform(
                    params['angle_range']['min'], params['angle_range']['max']
                ))
                self.uniform_time_factor = float(self.rng.uniform(
                    params['time_factor_range']['min'], params['time_factor_range']['max']
                ))
            angle: float = self.uniform_random_angle
            if self._time_varying:
                angle += self.uniform_time_factor * t
            u: float = self.velocity * np.cos(angle)
            v: float = self.velocity * np.sin(angle)
            return self._vec(u, v)
        except Exception as e:
            print(f"[ERROR] CurrentField._uniformRandomCurrent failed: {e}")
            raise

    def _sinusoidalRandomCurrent(self, x: float, y: float, t: float) -> np.ndarray:
        """Compute a sinusoidal current with random parameters."""
        try:
            if self.sinusoidal_random_periods is None:
                params: Dict = self.config['sinusoidal']
                self.sinusoidal_random_periods = self.rng.uniform(params['periods_range']['min'], params['periods_range']['max'], size=2)
                self.sinusoidal_random_gains = self.rng.uniform(params['gains_range']['min'], params['gains_range']['max'], size=2)
                self.sinusoidal_random_phases = self.rng.uniform(params['phases_range']['min'], params['phases_range']['max'], size=2)
                self.sinusoidal_time_freqs = self.rng.uniform(params['time_freq_range']['min'], params['time_freq_range']['max'], size=2)
            
            gain1: float; gain2: float
            gain1, gain2 = self.sinusoidal_random_gains
            period1: float; period2: float
            period1, period2 = self.sinusoidal_random_periods
            phase1: float; phase2: float
            phase1, phase2 = self.sinusoidal_random_phases
            
            if self._time_varying:
                phase1 += self.sinusoidal_time_freqs[0] * t
                phase2 += self.sinusoidal_time_freqs[1] * t

            safe_period1: float = period1 if abs(period1) > self.SAFE_EPSILON else self.SAFE_EPSILON
            safe_period2: float = period2 if abs(period2) > self.SAFE_EPSILON else self.SAFE_EPSILON
            
            u: Any; v: Any
            if self.library_type is None:
                u = -(1.0 - gain1) * self.velocity + gain1 * self.velocity * np.sin((2.0 * np.pi * y / safe_period1) + phase1)
                v = -(1.0 - gain2) * self.velocity + gain2 * self.velocity * np.cos((2.0 * np.pi * x / safe_period2) + phase2)
            else:
                u = -(1.0 - gain1) * self.velocity + gain1 * self.velocity * pyo.sin((2.0 * np.pi * y / safe_period1) + phase1)
                v = -(1.0 - gain2) * self.velocity + gain2 * self.velocity * pyo.cos((2.0 * np.pi * x / safe_period2) + phase2)
            return self._vec(u, v)
        except Exception as e:
            print(f"[ERROR] CurrentField._sinusoidalRandomCurrent failed: {e}")
            raise
            
    def _vortexRandomCurrent(self, x: float, y: float, t: float) -> np.ndarray:
        """Compute a vortex-based current with random parameters."""
        try:
            if self.vortex_random_number is None:
                params: Dict = self.config['vortex']
                self.vortex_random_number = int(self.rng.integers(params['number_range']['min'], params['number_range']['max']))
                self.vortex_random_centers = self.rng.uniform(-self.scenario_size_x, self.scenario_size_x, size=(self.vortex_random_number, 2))
                self.vortex_random_gains = self.rng.uniform(-0.1 * self.scenario_size_x, 0.1 * self.scenario_size_y, size=(self.vortex_random_number, 2))
                self.vortex_time_velocities = self.rng.uniform(params['time_velocity_range']['min'], params['time_velocity_range']['max'], size=(self.vortex_random_number, 2))
            
            wx: float = 0.0
            wy: float = 0.0
            for index in range(self.vortex_random_number):
                cx: float; cy: float
                cx, cy = self.vortex_random_centers[index]
                if self._time_varying:
                    cx += self.vortex_time_velocities[index, 0] * t
                    cy += self.vortex_time_velocities[index, 1] * t
                
                gainx: float; gainy: float
                gainx, gainy = self.vortex_random_gains[index]
                dx: float = x - cx
                dy: float = y - cy
                r_sq: float = dx * dx + dy * dy + self.SAFE_EPSILON
                wx += gainx * self.velocity * dy / r_sq
                wy += gainy * self.velocity * dx / r_sq
            return self._vec(wx, wy)
        except Exception as e:
            print(f"[ERROR] CurrentField._vortexRandomCurrent failed: {e}")
            raise

    def _logarithmicRandomFlowCurrent(self, x: float, y: float, t: float) -> np.ndarray:
        """Compute a logarithmic spiral flow with random parameters."""
        try:
            if self.logarithmic_random_number is None:
                params: Dict = self.config['logarithmic']
                self.logarithmic_random_number = int(self.rng.integers(params['number_range']['min'], params['number_range']['max']))
                self.logarithmic_random_rotations = self.rng.choice([True, False], size=self.logarithmic_random_number)
                self.logarithmic_random_centers = self.rng.uniform(-self.scenario_size_x, self.scenario_size_y, size=(self.logarithmic_random_number, 2))
            
            wx: float = 0.0
            wy: float = 0.0
            spiral_angle: float = np.pi / 4
            num_items: int = max(1, self.logarithmic_random_number)

            for index in range(self.logarithmic_random_number):
                cx: float; cy: float
                cx, cy = self.logarithmic_random_centers[index]
                rotation: bool = self.logarithmic_random_rotations[index]
                dx: float = x - cx
                dy: float = y - cy

                r: Any; sign: float; wx_radial: Any; wy_radial: Any; wx_tan: Any; wy_tan: Any
                if self.library_type is None:
                    r = np.sqrt(dx * dx + dy * dy) + self.SAFE_EPSILON
                    sign = -1.0 if rotation else 1.0
                    wx_radial = sign * self.velocity * dx / (r * num_items)
                    wy_radial = sign * self.velocity * dy / (r * num_items)
                    wx_tan = -self.velocity * dy / (r * num_items)
                    wy_tan = self.velocity * dx / (r * num_items)
                    wx += wx_radial * np.cos(spiral_angle) - wx_tan * np.sin(spiral_angle)
                    wy += wy_radial * np.cos(spiral_angle) - wy_tan * np.sin(spiral_angle)
                else:
                    r = pyo.sqrt(dx * dx + dy * dy) + self.SAFE_EPSILON
                    sign = -1.0 if rotation else 1.0
                    wx_radial = sign * self.velocity * dx / (r * num_items)
                    wy_radial = sign * self.velocity * dy / (r * num_items)
                    wx_tan = -self.velocity * dy / (r * num_items)
                    wy_tan = self.velocity * dx / (r * num_items)
                    wx += wx_radial * pyo.cos(spiral_angle) - wx_tan * pyo.sin(spiral_angle)
                    wy += wy_radial * pyo.cos(spiral_angle) - wy_tan * pyo.sin(spiral_angle)
            return self._vec(wx, wy)
        except Exception as e:
            print(f"[ERROR] CurrentField._logarithmicRandomFlowCurrent failed: {e}")
            raise
    
    def _gaussianSwirlRandomCurrent(self, x: float, y: float, t: float) -> np.ndarray:
        """Compute a Gaussian swirl field with random parameters."""
        try:
            if self.gaussian_swirl_random_number is None:
                params: Dict = self.config['gaussianSwir']
                self.gaussian_swirl_random_number = int(self.rng.integers(params['number_range']['min'], params['number_range']['max']))
                self.gaussian_swirl_random_centers = self.rng.uniform(-self.scenario_size_x / 2.0, self.scenario_size_y / 2.0, size=(self.gaussian_swirl_random_number, 2))
                self.gaussian_swirl_random_scales = self.rng.uniform(params['scales_range']['min'], 0.5 * self.scenario_size_x, size=self.gaussian_swirl_random_number)
                self.gaussian_swirl_random_gains = self.rng.uniform(params['gains_range']['min'], params['gains_range']['max'], size=self.gaussian_swirl_random_number)
                self.gaussian_swirl_time_velocities = self.rng.uniform(params['time_velocity_range']['min'], params['time_velocity_range']['max'], size=(self.gaussian_swirl_random_number, 2))
            
            wx: float = 0.0
            wy: float = 0.0
            for i in range(self.gaussian_swirl_random_number):
                cx: float; cy: float
                cx, cy = self.gaussian_swirl_random_centers[i]
                if self._time_varying:
                    cx += self.gaussian_swirl_time_velocities[i, 0] * t
                    cy += self.gaussian_swirl_time_velocities[i, 1] * t
                
                dx: float = x - cx
                dy: float = y - cy
                gain: float = self.gaussian_swirl_random_gains[i]
                r2: float = dx * dx + dy * dy
                scale_sq: float = self.gaussian_swirl_random_scales[i] ** 2
                safe_denominator: float = 2 * scale_sq + self.SAFE_EPSILON
                
                decay: Any; r: Any
                if self.library_type is None:
                    decay = np.exp(-r2 / safe_denominator)
                    r = np.sqrt(r2 + self.SAFE_EPSILON)
                    wx += gain * self.velocity * dy / r * decay
                    wy += -gain * self.velocity * dx / r * decay
                else:
                    decay = pyo.exp(-r2 / safe_denominator)
                    r = pyo.sqrt(r2 + self.SAFE_EPSILON)
                    wx += gain * self.velocity * dy / r * decay
                    wy += -gain * self.velocity * dx / r * decay
            return self._vec(wx, wy)
        except Exception as e:
            print(f"[ERROR] CurrentField._gaussianSwirlRandomCurrent failed: {e}")
            raise
    
    def _doubleGyreCurrent(self, x: float, y: float, t: float) -> np.ndarray:
        """Compute a double-gyre current field."""
        try:
            if self.double_gyre_A is None:
                params: Dict = self.config['doubleGyre']
                self.double_gyre_A = params['A_factor']['value'] * self.velocity
                self.double_gyre_eps = float(self.rng.uniform(params['epsilon_range']['min'], params['epsilon_range']['max']))
                self.double_gyre_time_freq = float(self.rng.uniform(params['time_freq_range']['min'], params['time_freq_range']['max']))
            
            omega: float = self.double_gyre_time_freq
            time_term: float = t if self._time_varying else 0.0
            a: float = self.double_gyre_eps * np.sin(omega * time_term)
            b: float = 1 - 2 * self.double_gyre_eps * np.sin(omega * time_term)
            
            X: float = (x + 0.5 * self.scenario_size_x) / (self.scenario_size_x + self.SAFE_EPSILON)
            Y: float = (y + 0.5 * self.scenario_size_y) / (self.scenario_size_y + self.SAFE_EPSILON)

            f: Any; dfdx: Any; u: Any; v: Any
            if self.library_type is None:
                X = np.clip(X, 0.0, 1.0)
                Y = np.clip(Y, 0.0, 1.0)
                f = a * X**2 + b * X
                dfdx = 2 * a * X + b
                u = -np.pi * self.double_gyre_A * np.sin(np.pi * f) * np.cos(np.pi * Y)
                v =  np.pi * self.double_gyre_A * np.cos(np.pi * f) * np.sin(np.pi * Y) * dfdx
            else:
                f = a * X**2 + b * X
                dfdx = 2 * a * X + b
                u = -np.pi * self.double_gyre_A * pyo.sin(np.pi * f) * pyo.cos(np.pi * Y)
                v =  np.pi * self.double_gyre_A * pyo.cos(np.pi * f) * pyo.sin(np.pi * Y) * dfdx
            return self._vec(u, v)
        except Exception as e:
            print(f"[ERROR] CurrentField._doubleGyreCurrent failed: {e}")
            raise

    def _coastalTidalCurrent(self, x: float, y: float, t: float) -> np.ndarray:
        """Compute a coastal tidal current field."""
        try:
            if self.tidal_coast_y0 is None:
                params: Dict = self.config['coastalTidal']
                self.tidal_coast_y0 = 0.0
                self.tidal_decay_scale = params['decay_scale_factor']['value'] * self.scenario_size_y
                self.tidal_period = float(self.rng.uniform(params['period_range']['min'], params['period_range']['max']))
                self.tidal_phase = float(self.rng.uniform(params['phase_range']['min'], params['phase_range']['max']))

            dyc: float = y - self.tidal_coast_y0
            time_arg: float = t if self._time_varying else x
            safe_decay_scale: float = self.tidal_decay_scale + self.SAFE_EPSILON
            safe_period: float = self.tidal_period + self.SAFE_EPSILON
            
            decay: Any; arg: Any; u: Any; v: Any
            if self.library_type is None:
                decay = np.exp(-np.abs(dyc) / safe_decay_scale)
                arg = 2 * np.pi * time_arg / safe_period + self.tidal_phase
                u = self.velocity * decay * np.cos(arg)
                v = 0.2 * self.velocity * decay * np.sin(arg)
            else:
                decay = pyo.exp(-abs(dyc) / safe_decay_scale)
                arg = 2 * np.pi * time_arg / safe_period + self.tidal_phase
                u = self.velocity * decay * pyo.cos(arg)
                v = 0.2 * self.velocity * decay * pyo.sin(arg)
            return self._vec(u, v)
        except Exception as e:
            print(f"[ERROR] CurrentField._coastalTidalCurrent failed: {e}")
            raise
    
    def _karmanVortexStreetCurrent(self, x: float, y: float, t: float) -> np.ndarray:
        """Compute a Karman vortex street current field."""
        try:
            if self.karman_spacing is None:
                params: Dict = self.config['karmanVortex']
                self.karman_spacing = float(self.rng.uniform(params['spacing_range']['min'], params['spacing_range']['max']))
                self.karman_strength = float(self.rng.uniform(params['strength_range']['min'], params['strength_range']['max']))
                self.karman_phase = float(self.rng.uniform(params['phase_range']['min'], params['phase_range']['max']))
                self.karman_wake_center = float(self.rng.uniform(params['wake_center_range']['min'], params['wake_center_range']['max']))
                self.karman_time_freq = float(self.rng.uniform(params['time_freq_range']['min'], params['time_freq_range']['max']))

            phase: float = self.karman_phase
            if self._time_varying:
                phase += self.karman_time_freq * t
            
            safe_spacing: float = self.karman_spacing + self.SAFE_EPSILON

            u: Any; v: Any
            if self.library_type is None:
                u = self.velocity * (1.0 - 0.15 * np.sin(2 * np.pi * (y - self.karman_wake_center) / safe_spacing + phase))
                v = self.karman_strength * self.velocity * np.sin(2 * np.pi * x / safe_spacing) * np.exp(-np.abs(y - self.karman_wake_center) / safe_spacing)
            else:
                u = self.velocity * (1.0 - 0.15 * pyo.sin(2 * np.pi * (y - self.karman_wake_center) / safe_spacing + phase))
                v = self.karman_strength * self.velocity * pyo.sin(2 * np.pi * x / safe_spacing) * pyo.exp(-abs(y - self.karman_wake_center) / safe_spacing)
            return self._vec(u, v)
        except Exception as e:
            print(f"[ERROR] CurrentField._karmanVortexStreetCurrent failed: {e}")
            raise

    def _linearShearCurrent(self, x: float, y: float, t: float) -> np.ndarray:
        """Compute a linear shear current field."""
        try:
            if self.shear_a is None:
                params: Dict = self.config['linearShear']
                scenario_max_size: float = max(self.scenario_size_x, self.scenario_size_y)
                a_base: float = float(self.rng.uniform(params['a_factor_range']['min'], params['a_factor_range']['max']))
                self.shear_a = a_base / (scenario_max_size + self.SAFE_EPSILON)
                self.shear_b = float(self.rng.uniform(params['b_range']['min'], params['b_range']['max']))
                self.shear_cross = float(self.rng.uniform(params['cross_range']['min'], params['cross_range']['max']))
            
            u: float = (self.shear_a * y + self.shear_b) * self.velocity
            v: float = self.shear_cross * self.velocity
            return self._vec(u, v)
        except Exception as e:
            print(f"[ERROR] CurrentField._linearShearCurrent failed: {e}")
            raise

    def _gaussianJetCurrent(self, x: float, y: float, t: float) -> np.ndarray:
        """Compute a Gaussian jet current field."""
        try:
            if self.jet_axis is None:
                params: Dict = self.config['gaussianJet']
                self.jet_axis = self.rng.choice(params['axis_choices']['values'])
                self.jet_sigma = float(self.rng.uniform(params['sigma_factor_range']['min'], params['sigma_factor_range']['max'])) * self.scenario_size_y
                self.jet_center_y = float(self.rng.uniform(params['center_factor_range']['min'], params['center_factor_range']['max'])) * self.scenario_size_y

            amp_factor: float = self.config['gaussianJet']['amplitude_factor']['value']
            safe_denominator: float = 2 * (self.jet_sigma**2) + self.SAFE_EPSILON
            
            u: Any; v: Any; amp: Any
            if self.jet_axis == "x":
                dy: float = y - self.jet_center_y
                if self.library_type is None:
                    amp = np.exp(-(dy * dy) / safe_denominator)
                    u, v = self.velocity * amp_factor * amp, 0.0
                else:
                    amp = pyo.exp(-(dy * dy) / safe_denominator)
                    u, v = self.velocity * amp_factor * amp, 0.0
                return self._vec(u, v)
            else: # axis is "y"
                dx: float = x - self.jet_center_y
                if self.library_type is None:
                    amp = np.exp(-(dx * dx) / safe_denominator)
                    u, v = 0.0, self.velocity * amp_factor * amp
                else:
                    amp = pyo.exp(-(dx * dx) / safe_denominator)
                    u, v = 0.0, self.velocity * amp_factor * amp
                return self._vec(u, v)
        except Exception as e:
            print(f"[ERROR] CurrentField._gaussianJetCurrent failed: {e}")
            raise

    def _riverOutflowCurrent(self, x: float, y: float, t: float) -> np.ndarray:
        """Compute a river outflow current field."""
        try:
            if self.river_mouth is None:
                params: Dict = self.config['riverOutflow']
                self.river_mouth = np.array([float(self.rng.uniform(params['mouth_x_factor_range']['min'], params['mouth_x_factor_range']['max'])) * self.scenario_size_x, params['mouth_y_factor']['value'] * self.scenario_size_y])
                self.river_spread = float(self.rng.uniform(params['spread_factor_range']['min'], params['spread_factor_range']['max'])) * self.scenario_size_x
                self.river_decay = float(self.rng.uniform(params['decay_range']['min'], params['decay_range']['max']))
            
            dx: float = x - self.river_mouth[0]
            dy: float = y - self.river_mouth[1]
            r2: float = dx * dx + dy * dy
            amp_spread_factor: float = self.config['riverOutflow']['spread_amplitude_factor']['value']

            safe_spread_denominator: float = 2 * (self.river_spread ** 2) + self.SAFE_EPSILON
            safe_r: Any = pyo.sqrt(r2 + self.SAFE_EPSILON) if self.library_type else np.sqrt(r2 + self.SAFE_EPSILON)
            
            amp_spread: Any; amp: Any; u: Any; v: Any
            if self.library_type is None:
                amp_spread = amp_spread_factor * np.exp(-((dx / safe_r)**2) * (self.scenario_size_x**2) / safe_spread_denominator)
                amp = (self.velocity * amp_spread) / (safe_r ** self.river_decay)
                u, v = amp * (dx / safe_r), amp * (dy / safe_r)
            else:
                amp_spread = amp_spread_factor * pyo.exp(-((dx / safe_r)**2) * (self.scenario_size_x**2) / safe_spread_denominator)
                amp = (self.velocity * amp_spread) / (safe_r ** self.river_decay)
                u, v = amp * (dx / safe_r), amp * (dy / safe_r)
            return self._vec(u, v)
        except Exception as e:
            print(f"[ERROR] CurrentField._riverOutflowCurrent failed: {e}")
            raise

    def _turbulenceNoiseCurrent(self, x: float, y: float, t: float) -> np.ndarray:
        """Compute a turbulence noise current field."""
        try:
            if self.turb_k is None:
                params: Dict = self.config['turbulenceNoise']
                m: int = int(self.rng.integers(params['modes_range']['min'], params['modes_range']['max']))
                safe_size: float = self.scenario_size_x + self.SAFE_EPSILON
                self.turb_k = self.rng.uniform(2 * np.pi / safe_size, 8 * np.pi / safe_size, size=(m, 2))
                self.turb_phases = self.rng.uniform(params['phases_range']['min'], params['phases_range']['max'], size=(m, 2))
                self.turb_gains = self.rng.uniform(params['gains_range']['min'], params['gains_range']['max'], size=m)
                self.turb_time_freqs = self.rng.uniform(params['time_freq_range']['min'], params['time_freq_range']['max'], size=(m, 2))
            
            u: float = 0.0
            v: float = 0.0
            for i in range(len(self.turb_gains)):
                kx: float; ky: float
                kx, ky = self.turb_k[i]
                phx: float; phy: float
                phx, phy = self.turb_phases[i]
                
                if self._time_varying:
                    phx += self.turb_time_freqs[i, 0] * t
                    phy += self.turb_time_freqs[i, 1] * t
                
                gain: float = self.turb_gains[i]
                
                if self.library_type is None:
                    u += gain * np.sin(kx * x + phx) * np.cos(ky * y + phy)
                    v += gain * np.cos(kx * x + phx) * np.sin(ky * y + phy)
                else:
                    u += gain * pyo.sin(kx * x + phx) * pyo.cos(ky * y + phy)
                    v += gain * pyo.cos(kx * x + phx) * pyo.sin(ky * y + phy)
            
            num_gains: int = max(1, len(self.turb_gains))
            return self._vec(self.velocity * u / num_gains, self.velocity * v / num_gains)
        except Exception as e:
            print(f"[ERROR] CurrentField._turbulenceNoiseCurrent failed: {e}")
            raise