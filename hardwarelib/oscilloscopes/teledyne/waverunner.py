"""Driver for Teledyne LeCroy WaveRunner oscilloscopes over VISA.

Tested on the WaveRunner 640Zi but should work with any MAUI-based
LeCroy scope that supports the standard 488.2 remote-control command
set (WaveRunner, WaveSurfer, HDO, etc.).

Waveform transfer uses the DEF9 binary block format with the WAVEDESC
header described in the MAUI Remote Control and Automation Manual.
"""

from __future__ import annotations

import struct
import time
from typing import Dict, Optional, Tuple

import numpy as np
import pyvisa

from hardwarelib.base import Oscilloscope

# ---------------------------------------------------------------------------
# WAVEDESC binary header layout (LECROY_2_3 template)
# Byte order is auto-detected at parse time from the header itself.
# ---------------------------------------------------------------------------

_WAVEDESC_FIELDS: list[dict] = [
    {"offset": 0,   "name": "DESCRIPTOR_NAME",  "fmt": "16s"},
    {"offset": 16,  "name": "TEMPLATE_NAME",    "fmt": "16s"},
    {"offset": 32,  "name": "COMM_TYPE",         "fmt": "h"},    # 0=byte, 1=word
    {"offset": 34,  "name": "COMM_ORDER",        "fmt": "h"},    # 0=HiFirst, 1=LoFirst
    {"offset": 36,  "name": "WAVE_DESCRIPTOR",   "fmt": "i"},
    {"offset": 40,  "name": "USER_TEXT",          "fmt": "i"},
    {"offset": 44,  "name": "RES_DESC1",          "fmt": "i"},
    {"offset": 48,  "name": "TRIGTIME_ARRAY",     "fmt": "i"},
    {"offset": 52,  "name": "RIS_TIME_ARRAY",     "fmt": "i"},
    {"offset": 56,  "name": "RES_ARRAY1",         "fmt": "i"},
    {"offset": 60,  "name": "WAVE_ARRAY_1",       "fmt": "i"},
    {"offset": 64,  "name": "WAVE_ARRAY_2",       "fmt": "i"},
    {"offset": 76,  "name": "INSTRUMENT_NAME",   "fmt": "16s"},
    {"offset": 92,  "name": "INSTRUMENT_NUMBER", "fmt": "I"},
    {"offset": 96,  "name": "TRACE_LABEL",       "fmt": "16s"},
    {"offset": 116, "name": "WAVE_ARRAY_COUNT",  "fmt": "i"},
    {"offset": 120, "name": "PNTS_PER_SCREEN",   "fmt": "i"},
    {"offset": 124, "name": "FIRST_VALID_PNT",   "fmt": "i"},
    {"offset": 128, "name": "LAST_VALID_PNT",    "fmt": "i"},
    {"offset": 132, "name": "FIRST_POINT",        "fmt": "i"},
    {"offset": 136, "name": "SPARSING_FACTOR",    "fmt": "i"},
    {"offset": 140, "name": "SEGMENT_INDEX",      "fmt": "i"},
    {"offset": 144, "name": "SUBARRAY_COUNT",     "fmt": "i"},
    {"offset": 148, "name": "SWEEPS_PER_ACQ",     "fmt": "i"},
    {"offset": 156, "name": "VERTICAL_GAIN",      "fmt": "f"},
    {"offset": 160, "name": "VERTICAL_OFFSET",    "fmt": "f"},
    {"offset": 164, "name": "MAX_VALUE",           "fmt": "f"},
    {"offset": 168, "name": "MIN_VALUE",           "fmt": "f"},
    {"offset": 172, "name": "NOMINAL_BITS",       "fmt": "h"},
    {"offset": 176, "name": "HORIZ_INTERVAL",     "fmt": "f"},
    {"offset": 180, "name": "HORIZ_OFFSET",       "fmt": "d"},
    {"offset": 188, "name": "PIXEL_OFFSET",       "fmt": "d"},
    {"offset": 196, "name": "VERTUNIT",           "fmt": "48s"},
    {"offset": 244, "name": "HORUNIT",            "fmt": "48s"},
    {"offset": 292, "name": "HORIZ_UNCERTAINTY",  "fmt": "f"},
    {"offset": 312, "name": "ACQ_DURATION",       "fmt": "f"},
    {"offset": 316, "name": "RECORD_TYPE",        "fmt": "h"},
    {"offset": 324, "name": "TIMEBASE",           "fmt": "h"},
    {"offset": 326, "name": "VERT_COUPLING",      "fmt": "h"},
    {"offset": 328, "name": "PROBE_ATT",          "fmt": "f"},
    {"offset": 344, "name": "WAVE_SOURCE",        "fmt": "h"},
]


def _detect_endian(buf: bytes) -> str:
    """Auto-detect byte order from the WAVEDESC block.

    The WAVE_DESCRIPTOR length at offset 36 should be a reasonable
    value (typically 346).  We read it in both endiannesses and pick
    the one that makes sense.
    """
    if len(buf) < 40:
        return ">"
    len_be = struct.unpack_from(">i", buf, 36)[0]
    len_le = struct.unpack_from("<i", buf, 36)[0]
    if 100 < len_be < 100_000:
        return ">"
    if 100 < len_le < 100_000:
        return "<"
    return ">"


def _parse_wavedesc(buf: bytes) -> tuple[Dict, str]:
    """Parse a WAVEDESC binary block into a dict of typed values.

    Returns ``(descriptor_dict, endian_prefix)`` where *endian_prefix*
    is ``">"`` (big) or ``"<"`` (little).
    """
    endian = _detect_endian(buf)
    desc: Dict = {}
    for field in _WAVEDESC_FIELDS:
        fmt = field["fmt"]
        if fmt[-1] != "s":
            fmt = endian + fmt
        size = struct.calcsize(fmt)
        raw = buf[field["offset"] : field["offset"] + size]
        if len(raw) < size:
            continue
        (val,) = struct.unpack(fmt, raw)
        if isinstance(val, bytes):
            val = val.split(b"\x00", 1)[0].decode("ascii", errors="replace")
        desc[field["name"]] = val
    return desc, endian


def _find_block_start(raw: bytes) -> int:
    """Locate the IEEE 488.2 ``#N...`` definite-length block header.

    Returns the index of the ``#`` character.  Raises ValueError when
    no valid header is found.
    """
    idx = raw.find(b"#")
    if idx < 0:
        raise ValueError("No IEEE 488.2 block header (#) found in response.")
    return idx


def _strip_block_header(raw: bytes) -> bytes:
    """Remove the ``#Ndddd...`` header and return the payload bytes."""
    idx = _find_block_start(raw)
    n_digits = int(raw[idx + 1 : idx + 2])
    payload_start = idx + 2 + n_digits
    return raw[payload_start:]


class LeCroyWaveRunner(Oscilloscope):
    """VISA driver for Teledyne LeCroy WaveRunner oscilloscopes.

    Parameters
    ----------
    resource_name : str
        VISA resource string, e.g.
        ``"TCPIP0::192.168.1.100::INSTR"`` (VXI-11) or
        ``"USB0::0x05ff::0x1023::SERIALNO::INSTR"``.
    timeout_ms : int
        VISA I/O timeout in milliseconds (default 30 s).
    """

    def __init__(self, resource_name: str, timeout_ms: int = 30_000):
        self.resource_name = resource_name
        self.timeout_ms = timeout_ms
        self._rm: Optional[pyvisa.ResourceManager] = None
        self._inst = None

    # -- connection ---------------------------------------------------------

    def open(self) -> None:
        self._rm = pyvisa.ResourceManager()
        self._inst = self._rm.open_resource(self.resource_name)
        self._inst.timeout = self.timeout_ms

        # LeCroy over USBTMC doesn't reliably use \n termination on
        # responses, so we must avoid read() blocking on a termchar
        # that never arrives.  Disable termchar-based reads and use
        # read_raw() everywhere instead (see query()).
        self._inst.read_termination = None
        self._inst.write_termination = "\n"

        if hasattr(self._inst, "chunk_size"):
            self._inst.chunk_size = 1024 * 1024

        try:
            self._inst.clear()
        except Exception:
            pass

        self.write("CHDR OFF")
        self.write("CORD HI")
        self.write("COMM_FORMAT DEF9,WORD,BIN")
        self.query("*OPC?")

    def close(self) -> None:
        try:
            if self._inst is not None:
                self._inst.close()
        finally:
            self._inst = None
            if self._rm is not None:
                self._rm.close()
                self._rm = None

    # -- low-level I/O ------------------------------------------------------

    def write(self, cmd: str) -> None:
        if self._inst is None:
            raise RuntimeError("Scope not open.")
        self._inst.write(cmd)

    def query(self, cmd: str) -> str:
        if self._inst is None:
            raise RuntimeError("Scope not open.")
        self.write(cmd)
        raw = self._inst.read_raw()
        return raw.decode("ascii", errors="replace").strip()

    def flush(self) -> None:
        """Clear VISA I/O buffers and reset the scope's status."""
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

    # -- Oscilloscope interface ---------------------------------------------

    def idn(self) -> str:
        return self.query("*IDN?")

    def acquire_single_and_wait(
        self, timeout_s: float, settle_s: float = 0.0
    ) -> None:
        self.write("TRMD SINGLE")
        t0 = time.time()
        last_status = ""
        while time.time() - t0 < timeout_s:
            status = self.query("TRIG_MODE?").upper().strip()
            last_status = status
            if status == "STOP":
                self.query("*OPC?")
                if settle_s > 0:
                    time.sleep(settle_s)
                return
            time.sleep(0.05)
        raise TimeoutError(
            f"Scope single acquisition timed out; last trigger status was {last_status!r}"
        )

    def get_timebase(self) -> float:
        return float(self.query("TDIV?"))

    def set_timebase(self, scale_s_per_div: float) -> None:
        self.write(f"TDIV {scale_s_per_div:.6E}")
        self.query("*OPC?")

    def set_trigger_mode(self, mode: str = "AUTO") -> None:
        mapping = {
            "AUTO": "AUTO",
            "NORMAL": "NORM",
            "NORM": "NORM",
            "SINGLE": "SINGLE",
            "STOP": "STOP",
        }
        cmd_mode = mapping.get(mode.upper(), mode.upper())
        self.write(f"TRMD {cmd_mode}")

    def read_waveform(
        self, channel: int
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        if channel not in (1, 2, 3, 4):
            raise ValueError("Scope channel must be 1..4")

        self.write("TRMD STOP")
        self.write(f"C{channel}:WF?")
        time.sleep(0.1)

        raw = self._inst.read_raw()

        payload = _strip_block_header(raw)

        desc, endian = _parse_wavedesc(payload)

        wave_desc_len = desc["WAVE_DESCRIPTOR"]
        user_text_len = desc.get("USER_TEXT", 0)
        trigtime_len = desc.get("TRIGTIME_ARRAY", 0)
        ristime_len = desc.get("RIS_TIME_ARRAY", 0)

        data_start = wave_desc_len + user_text_len + trigtime_len + ristime_len
        data_len = desc["WAVE_ARRAY_1"]
        data_bytes = payload[data_start : data_start + data_len]

        comm_type = desc.get("COMM_TYPE", 1)
        if comm_type == 0:
            adc = np.frombuffer(data_bytes, dtype=np.int8).astype(np.float64)
        else:
            dt = ">i2" if endian == ">" else "<i2"
            adc = np.frombuffer(data_bytes, dtype=dt).astype(np.float64)

        vgain = desc["VERTICAL_GAIN"]
        voff = desc["VERTICAL_OFFSET"]
        y_v = adc * vgain - voff

        hi = desc["HORIZ_INTERVAL"]
        ho = desc["HORIZ_OFFSET"]
        n_pts = len(adc)
        t_s = np.arange(n_pts) * hi + ho

        meta: Dict[str, float] = {
            "vertical_gain": vgain,
            "vertical_offset": voff,
            "horiz_interval": hi,
            "horiz_offset": ho,
            "wave_array_count": float(desc.get("WAVE_ARRAY_COUNT", n_pts)),
            "nominal_bits": float(desc.get("NOMINAL_BITS", 0)),
            "probe_att": float(desc.get("PROBE_ATT", 1.0)),
            "comm_type": float(comm_type),
            "endian": 0.0 if endian == ">" else 1.0,
        }
        return t_s, y_v, meta
