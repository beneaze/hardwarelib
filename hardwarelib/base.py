"""Abstract base classes for lab hardware."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np


class Instrument(ABC):
    """Base class for all lab instruments."""

    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class Oscilloscope(Instrument):
    """Abstract oscilloscope interface."""

    @abstractmethod
    def idn(self) -> str: ...

    @abstractmethod
    def acquire_single_and_wait(
        self, timeout_s: float, settle_s: float = 0.0
    ) -> None: ...

    @abstractmethod
    def read_waveform(
        self, channel: int
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """Return (time_s, voltage_v, metadata)."""
        ...


class SignalGenerator(Instrument):
    """Abstract RF / microwave signal generator interface."""

    @abstractmethod
    def set_output(self, enabled: bool) -> None: ...

    @abstractmethod
    def apply(
        self,
        freq_hz: float,
        power_dbm: Optional[float] = None,
        enabled: Optional[bool] = None,
    ) -> None: ...
