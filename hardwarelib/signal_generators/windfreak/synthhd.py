"""Driver for the Windfreak SynthHD dual-channel RF synthesizer over serial."""

from __future__ import annotations

from typing import Optional

import serial

from hardwarelib.base import SignalGenerator


class WindfreakSynthHD(SignalGenerator):
    """Serial driver for the Windfreak SynthHD.

    Parameters
    ----------
    port : str
        Serial port name, e.g. ``"COM12"`` or ``"/dev/ttyUSB0"``.
    channel : int
        Default RF channel (0 or 1).
    timeout_s : float
        Serial read/write timeout in seconds.
    """

    def __init__(self, port: str, channel: int = 0, timeout_s: float = 0.25):
        self.port = port
        self.channel = channel
        self.timeout_s = timeout_s
        self._ser: Optional[serial.Serial] = None

    def open(self) -> None:
        self._ser = serial.Serial(
            port=self.port,
            baudrate=9600,
            timeout=self.timeout_s,
            write_timeout=self.timeout_s,
        )
        self._ser.reset_input_buffer()
        self._ser.reset_output_buffer()
        self.select_channel(self.channel)

    def close(self) -> None:
        if self._ser is not None:
            self._ser.close()
            self._ser = None

    def _write_packet(self, packet: str) -> None:
        if self._ser is None:
            raise RuntimeError("Windfreak serial port not open.")
        self._ser.write(packet.encode("ascii"))
        self._ser.flush()

    def select_channel(self, channel: int) -> None:
        if channel not in (0, 1):
            raise ValueError("Windfreak channel must be 0 or 1.")
        self.channel = channel
        self._write_packet(f"C{channel}")

    # -- SignalGenerator interface ---------------------------------------------

    def set_output(self, enabled: bool) -> None:
        self._write_packet("E1r1" if enabled else "E0r0")

    def apply(
        self,
        freq_hz: float,
        power_dbm: Optional[float] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        pieces = [f"C{self.channel}", f"f{freq_hz / 1e6:.8f}"]
        if power_dbm is not None:
            pieces.append(f"W{power_dbm:.3f}")
        if enabled is not None:
            pieces.append("E1r1" if enabled else "E0r0")
        self._write_packet("".join(pieces))
