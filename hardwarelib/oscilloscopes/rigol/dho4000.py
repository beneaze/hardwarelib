"""Driver for Rigol DHO/HDO 4000-series oscilloscopes over VISA (SCPI)."""

from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

import numpy as np
import pyvisa

from hardwarelib.base import Oscilloscope


def _parse_tmc_block(raw: bytes) -> bytes:
    """Decode a TMC (IEEE 488.2) definite-length block header."""
    if not raw:
        raise ValueError("Empty block returned by scope.")
    if raw[:1] != b"#":
        raise ValueError(f"Unexpected TMC header: {raw[:20]!r}")
    ndigits = int(raw[1:2].decode())
    nbytes = int(raw[2 : 2 + ndigits].decode())
    start = 2 + ndigits
    end = start + nbytes
    if len(raw) < end:
        raise ValueError(
            f"Incomplete block: expected {nbytes} data bytes, got {len(raw) - start}"
        )
    return raw[start:end]


class RigolDHO4000(Oscilloscope):
    """VISA driver for the Rigol DHO/HDO 4000 oscilloscope family.

    Parameters
    ----------
    resource_name : str
        VISA resource string, e.g. ``"TCPIP0::192.168.1.50::INSTR"``.
    timeout_ms : int
        VISA I/O timeout in milliseconds.
    """

    def __init__(self, resource_name: str, timeout_ms: int = 15_000):
        self.resource_name = resource_name
        self.timeout_ms = timeout_ms
        self._rm: Optional[pyvisa.ResourceManager] = None
        self._inst = None

    def open(self) -> None:
        self._rm = pyvisa.ResourceManager()
        self._inst = self._rm.open_resource(self.resource_name)
        self._inst.timeout = self.timeout_ms
        self._inst.write_termination = "\n"
        self._inst.read_termination = "\n"

    def close(self) -> None:
        try:
            if self._inst is not None:
                self._inst.close()
        finally:
            self._inst = None
            if self._rm is not None:
                self._rm.close()
                self._rm = None

    def write(self, cmd: str) -> None:
        if self._inst is None:
            raise RuntimeError("Scope not open.")
        self._inst.write(cmd)

    def query(self, cmd: str) -> str:
        if self._inst is None:
            raise RuntimeError("Scope not open.")
        return self._inst.query(cmd).strip()

    def query_binary_block(self, cmd: str) -> bytes:
        if self._inst is None:
            raise RuntimeError("Scope not open.")
        self._inst.write(cmd)
        raw = self._inst.read_raw()
        return _parse_tmc_block(raw)

    def flush(self) -> None:
        """Clear the VISA I/O buffers and reset the scope's status."""
        if self._inst is None:
            return
        try:
            self._inst.clear()
        except Exception:
            pass
        try:
            self.write("*CLS")
        except Exception:
            pass

    # -- Oscilloscope interface ------------------------------------------------

    def idn(self) -> str:
        return self.query("*IDN?")

    def acquire_single_and_wait(
        self, timeout_s: float, settle_s: float = 0.0
    ) -> None:
        self.write(":SINGle")
        t0 = time.time()
        last_status = ""
        while time.time() - t0 < timeout_s:
            status = self.query(":TRIGger:STATus?").upper()
            last_status = status
            if status == "STOP":
                if settle_s > 0:
                    time.sleep(settle_s)
                return
            time.sleep(0.05)
        raise TimeoutError(
            f"Scope single acquisition timed out; last trigger status was {last_status!r}"
        )

    def get_timebase(self) -> float:
        return float(self.query(":TIMebase:MAIN:SCALe?"))

    def set_timebase(self, scale_s_per_div: float) -> None:
        self.write(f":TIMebase:MAIN:SCALe {scale_s_per_div:.6E}")

    def set_trigger_mode(self, mode: str = "AUTO") -> None:
        self.write(f":TRIGger:SWEep {mode}")

    def read_waveform(
        self, channel: int
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        if channel not in (1, 2, 3, 4):
            raise ValueError("Scope channel must be 1..4")

        self.write(":STOP")
        self.write(f":WAVeform:SOURce CHANnel{channel}")
        self.write(":WAVeform:MODE RAW")
        self.write(":WAVeform:FORMat BYTE")

        points = int(float(self.query(":WAVeform:POINts?")))
        self.write(":WAVeform:STARt 1")
        self.write(f":WAVeform:STOP {points}")

        pre = self.query(":WAVeform:PREamble?")
        vals = [float(x) for x in pre.split(",")]
        if len(vals) != 10:
            raise ValueError(f"Unexpected preamble: {pre}")

        (
            fmt,
            dtype_mode,
            points_from_pre,
            count,
            xincrement,
            xorigin,
            xreference,
            yincrement,
            yorigin,
            yreference,
        ) = vals

        raw = self.query_binary_block(":WAVeform:DATA?")
        adc = np.frombuffer(raw, dtype=np.uint8).astype(np.float64)
        y_v = (adc - yorigin - yreference) * yincrement
        t_s = xorigin + (np.arange(adc.size) - xreference) * xincrement

        meta = {
            "format": fmt,
            "type": dtype_mode,
            "points": points_from_pre,
            "count": count,
            "xincrement": xincrement,
            "xorigin": xorigin,
            "xreference": xreference,
            "yincrement": yincrement,
            "yorigin": yorigin,
            "yreference": yreference,
        }
        return t_s, y_v, meta
