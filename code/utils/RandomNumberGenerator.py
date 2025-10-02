# ===============================================================================
# RandomNumberGenerator Utility
#
# Author: José Antonio González Prieto
# Date: 01/11/2025
# Version: 1.0
# Description:
#   Reproducible wrapper around NumPy's Generator. Provides a master RNG that
#   can spawn child generators with independent seeds. Public methods follow
#   camelCase naming; internal variables use snake_case. All public methods
#   include basic error handling that logs to stdout and re-raises to stop
#   execution on failure.
# ===============================================================================
from __future__ import annotations

# =======================================================================
# IMPORTS
# =======================================================================
from typing import Any, Dict, List, Optional
import numpy as np


class RandomNumberGenerator:
    """
    Reproducible random number generator built on top of NumPy's `default_rng`.

    Notes
    -----
    - If no `seed` is provided, a `SeedSequence` is created and its entropy is
      used as the seed to initialize the internal Generator.
    - Use :meth:`createGenerators` to derive independent child generators.

    Attributes
    ----------
    seed : int
        Seed used to initialize the internal Generator.
    rng : np.random.Generator
        The underlying NumPy Generator instance.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        """Initialize the RandomNumberGenerator.

        Parameters
        ----------
        seed : int, optional
            Seed for reproducibility. If None, a SeedSequence-derived value is used.

        Raises
        ------
        Exception
            If initialization fails for any reason.
        """
        try:
            if seed is None:
                seed_sequence: np.random.SeedSequence = np.random.SeedSequence()
                self.seed: int = int(seed_sequence.entropy)
            else:
                self.seed = int(seed)

            self.rng: np.random.Generator = np.random.default_rng(self.seed)
        except Exception as e:
            print(f"[ERROR] RandomNumberGenerator.__init__ failed: {e}")
            raise

    def integers(
        self,
        low: int,
        high: Optional[int] = None,
        size: int | tuple[int, ...] = 1,
        endpoint: bool = False,
        dtype: Any = np.int64,
    ) -> np.ndarray | int:
        """Draw random integers from `low` (inclusive) to `high` (exclusive by default).

        Parameters
        ----------
        low : int
            Lowest (signed) integer to be drawn from the distribution (inclusive).
        high : int, optional
            If provided, one above the largest (signed) integer to be drawn.
            If None, values are drawn from [0, low) instead.
        size : int | tuple[int, ...], optional
            Output shape. Default is 1.
        endpoint : bool, optional
            If True, sample from [low, high] instead of [low, high). Default False.
        dtype : Any, optional
            Desired dtype of the result. Default `np.int64`.

        Returns
        -------
        np.ndarray | int
            Random integers of shape `size`.

        Raises
        ------
        Exception
            If sampling fails.
        """
        try:
            return self.rng.integers(low, high, size, endpoint=endpoint, dtype=dtype)
        except Exception as e:
            print(f"[ERROR] RandomNumberGenerator.integers failed: {e}")
            raise

    def uniform(
        self,
        low: float = 0.0,
        high: float = 1.0,
        size: int | tuple[int, ...] = 1,
        dtype: Any = float,
    ) -> np.ndarray | float:
        """Draw samples from a uniform distribution over [low, high).

        Parameters
        ----------
        low : float, optional
            Lower boundary of the output interval. Default 0.0.
        high : float, optional
            Upper boundary of the output interval. Default 1.0.
        size : int | tuple[int, ...], optional
            Output shape. Default 1.
        dtype : Any, optional
            Desired dtype of the result. Default float.

        Returns
        -------
        np.ndarray | float
            Samples from the uniform distribution.

        Raises
        ------
        Exception
            If sampling fails.
        """
        try:
            vals: np.ndarray | float = self.rng.uniform(low, high, size)
            return vals.astype(dtype) if isinstance(vals, np.ndarray) else dtype(vals)
        except Exception as e:
            print(f"[ERROR] RandomNumberGenerator.uniform failed: {e}")
            raise

    def choice(
        self,
        a: Any,
        size: Optional[int | tuple[int, ...]] = 1,
        replace: bool = True,
        p: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Generate a random sample from a given 1-D array-like object.

        Parameters
        ----------
        a : array_like
            If an ndarray, a random sample is generated from its elements.
            If an int, the random sample is generated as if `a` were `np.arange(a)`.
        size : int | tuple[int, ...] | None, optional
            Output shape. If None, a single value is returned. Default 1.
        replace : bool, optional
            Whether the sample is with or without replacement. Default True.
        p : np.ndarray, optional
            The probabilities associated with each entry in `a`. If not given,
            a uniform distribution is assumed.

        Returns
        -------
        np.ndarray
            Sampled values with shape `size`.

        Raises
        ------
        Exception
            If sampling fails.
        """
        try:
            return self.rng.choice(a, size=size, replace=replace, p=p)
        except Exception as e:
            print(f"[ERROR] RandomNumberGenerator.choice failed: {e}")
            raise

    def createGenerators(self, identifiers: List[str]) -> Dict[str, "RandomNumberGenerator"]:
        """Create a dictionary of independent child generators.

        Parameters
        ----------
        identifiers : list[str]
            List of string identifiers to name each child generator.

        Returns
        -------
        dict[str, RandomNumberGenerator]
            A mapping from identifier to a new, independently seeded
            `RandomNumberGenerator` instance.

        Raises
        ------
        Exception
            If creation of any child generator fails.
        """
        try:
            sub_generators: Dict[str, RandomNumberGenerator] = {}
            for identifier_str in identifiers:
                new_seed: int = int(self.rng.integers(low=1, high=int(1e12)))
                sub_generators[identifier_str] = RandomNumberGenerator(seed=new_seed)
            return sub_generators
        except Exception as e:
            print(f"[ERROR] RandomNumberGenerator.createGenerators failed: {e}")
            raise