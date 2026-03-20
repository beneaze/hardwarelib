"""Driver for Rigol RSA3000E-series real-time spectrum analyzers over VISA (SCPI).

Tested on the RSA3030E (9 kHz – 3 GHz).  Should work with other RSA3000E
variants (RSA3015E, -TG models) without modification.

NOTE: Rigol SAs do NOT properly implement ``*OPC?`` for sweep synchronisation.
The query returns "1" immediately regardless of sweep progress.  This driver
works around the issue by querying the expected sweep time and sleeping for
that duration, then verifying the trace data actually updated.
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Optional, Tuple

import numpy as np
import pyvisa

log = logging.getLogger(__name__)

from hardwarelib.base import SpectrumAnalyzer


class RigolRSA3000E(SpectrumAnalyzer):
    """VISA driver for the Rigol RSA3000E spectrum analyzer family.

    Operates in GPSA (General Purpose Spectrum Analyzer) mode.

    Parameters
    ----------
    resource_name : str
        VISA resource string, e.g.
        ``"USB0::0x1AB1::0x0960::RSA3E243900001::INSTR"`` or
        ``"TCPIP0::192.168.1.51::INSTR"``.
    timeout_ms : int
        VISA I/O timeout in milliseconds.
    """

    def __init__(self, resource_name: str, timeout_ms: int = 30_000):
        self.resource_name = resource_name
        self.timeout_ms = timeout_ms
        self._rm: Optional[pyvisa.ResourceManager] = None
        self._inst = None
        self._last_trace_hash: Optional[bytes] = None

    # -- Instrument lifecycle --------------------------------------------------

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

    # -- Low-level SCPI helpers ------------------------------------------------

    def write(self, cmd: str) -> None:
        if self._inst is None:
            raise RuntimeError("Spectrum analyzer not open.")
        self._inst.write(cmd)

    def query(self, cmd: str) -> str:
        if self._inst is None:
            raise RuntimeError("Spectrum analyzer not open.")
        return self._inst.query(cmd).strip()

    def flush(self) -> None:
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

    def reset(self) -> None:
        self.write("*RST")
        self.write("*CLS")
        time.sleep(5.0)

    def _opc_wait(self, timeout_s: float = 10.0) -> None:
        """Send ``*OPC?`` — unreliable on Rigol, kept for compatibility only."""
        saved = self._inst.timeout
        self._inst.timeout = int(timeout_s * 1000)
        try:
            self.query("*OPC?")
        finally:
            self._inst.timeout = saved

    # -- Write-then-verify helper ----------------------------------------------

    def _set_and_verify(
        self,
        write_cmd: str,
        query_cmd: str,
        value: float,
        tol_frac: float = 0.01,
        retries: int = 3,
        delay: float = 0.15,
    ) -> float:
        """Write a numeric parameter and read it back to confirm it took effect.

        Retries up to *retries* times with *delay* seconds between attempts.
        Returns the value as read back from the instrument.
        Raises ``RuntimeError`` if the value still doesn't match after all retries.
        """
        for attempt in range(retries):
            self.write(write_cmd)
            time.sleep(delay)
            readback = float(self.query(query_cmd))
            if value == 0:
                if readback == 0:
                    return readback
            elif abs(readback - value) / abs(value) <= tol_frac:
                return readback
        raise RuntimeError(
            f"SA did not accept setting after {retries} attempts: "
            f"sent {write_cmd!r}, expected {value}, got {readback}"
        )

    # -- SpectrumAnalyzer interface --------------------------------------------

    def idn(self) -> str:
        return self.query("*IDN?")

    def set_center_frequency(self, freq_hz: float) -> None:
        self.write(f":SENSe:FREQuency:CENTer {freq_hz:.0f}")

    def get_center_frequency(self) -> float:
        return float(self.query(":SENSe:FREQuency:CENTer?"))

    def set_span(self, span_hz: float) -> None:
        self.write(f":SENSe:FREQuency:SPAN {span_hz:.0f}")

    def get_span(self) -> float:
        return float(self.query(":SENSe:FREQuency:SPAN?"))

    def set_start_frequency(self, freq_hz: float) -> None:
        self.write(f":SENSe:FREQuency:STARt {freq_hz:.0f}")

    def get_start_frequency(self) -> float:
        return float(self.query(":SENSe:FREQuency:STARt?"))

    def set_stop_frequency(self, freq_hz: float) -> None:
        self.write(f":SENSe:FREQuency:STOP {freq_hz:.0f}")

    def get_stop_frequency(self) -> float:
        return float(self.query(":SENSe:FREQuency:STOP?"))

    def set_rbw(self, rbw_hz: float) -> None:
        self.write(":SENSe:BANDwidth:RESolution:AUTO OFF")
        self.write(f":SENSe:BANDwidth:RESolution {rbw_hz:.0f}")

    def get_rbw(self) -> float:
        return float(self.query(":SENSe:BANDwidth:RESolution?"))

    def set_rbw_auto(self) -> None:
        self.write(":SENSe:BANDwidth:RESolution:AUTO ON")

    def set_vbw(self, vbw_hz: float) -> None:
        self.write(":SENSe:BANDwidth:VIDeo:AUTO OFF")
        self.write(f":SENSe:BANDwidth:VIDeo {vbw_hz:.0f}")

    def get_vbw(self) -> float:
        return float(self.query(":SENSe:BANDwidth:VIDeo?"))

    def set_vbw_auto(self) -> None:
        self.write(":SENSe:BANDwidth:VIDeo:AUTO ON")

    def set_reference_level(self, level_dbm: float) -> None:
        self.write(f":DISPlay:WINDow:TRACe:Y:SCALe:RLEVel {level_dbm:.1f}")

    def get_reference_level(self) -> float:
        return float(self.query(":DISPlay:WINDow:TRACe:Y:SCALe:RLEVel?"))

    def set_input_attenuation(self, atten_db: float) -> None:
        self.write(":SENSe:POWer:RF:ATTenuation:AUTO OFF")
        self.write(f":SENSe:POWer:RF:ATTenuation {atten_db:.0f}")

    def get_input_attenuation(self) -> float:
        """Return current RF input attenuation in dB."""
        return float(self.query(":SENSe:POWer:RF:ATTenuation?"))

    def is_input_attenuation_auto(self) -> bool:
        """Return whether auto-attenuation is currently enabled."""
        return self.query(":SENSe:POWer:RF:ATTenuation:AUTO?").upper() in ("1", "ON")

    def set_input_attenuation_auto(self) -> None:
        self.write(":SENSe:POWer:RF:ATTenuation:AUTO ON")

    def _apply_ref_and_atten(
        self,
        ref_level_dbm: float,
        atten_offset_db: float = 10.0,
    ) -> None:
        """Set reference level and matching manual attenuation in sync.

        Sets attenuation to ``ref_level_dbm + atten_offset_db`` (clamped
        to [0, 50] dB).  Manual mode bypasses auto-attenuation entirely.
        Prefer :meth:`_apply_ref_auto` for normal measurement workflows.
        """
        atten = max(0.0, min(ref_level_dbm + atten_offset_db, 50.0))
        self.write(":SENSe:POWer:RF:ATTenuation:AUTO OFF")
        self.write(f":SENSe:POWer:RF:ATTenuation {atten:.0f}")
        self.write(f":DISPlay:WINDow:TRACe:Y:SCALe:RLEVel {ref_level_dbm:.1f}")

    def _apply_ref_auto(self, ref_level_dbm: float) -> None:
        """Enable auto-attenuation, set reference level, and wait for settle.

        Auto-attenuation couples the internal attenuator to the reference
        level.  After changing the reference level we must wait for the SA
        to recalculate and apply the new attenuation before triggering a
        sweep — this is the settle time that continuous-sweep mode gets
        for free between sweeps but single-trigger mode does not.
        """
        self.set_input_attenuation_auto()
        self.set_reference_level(ref_level_dbm)
        self._settle_auto_attenuation()

    def _settle_auto_attenuation(
        self,
        poll_interval_s: float = 0.15,
        max_wait_s: float = 1.0,
    ) -> float:
        """Wait until the auto-attenuation value has stabilised.

        Polls the attenuation read-back at *poll_interval_s* intervals.
        Returns once two consecutive reads agree, or after *max_wait_s*
        total.  Returns the settled attenuation value in dB.
        """
        prev = self.get_input_attenuation()
        waited = 0.0
        while waited < max_wait_s:
            time.sleep(poll_interval_s)
            waited += poll_interval_s
            current = self.get_input_attenuation()
            if current == prev:
                return current
            prev = current
        return current

    # -- Clipping / overload detection -----------------------------------------

    def is_input_overloaded(self) -> bool:
        """Query the questionable-power status register for an overload condition.

        Bit 3 (value 8) of ``:STATus:QUEStionable:CONDition?`` signals an
        IF / ADC overload on the RSA3000E.  Returns ``True`` when the
        instrument reports overload.
        """
        raw = int(self.query(":STATus:QUEStionable:CONDition?"))
        return bool(raw & 0x08)

    def check_clipping(
        self,
        trace: int = 1,
        margin_db: float = 1.0,
    ) -> dict:
        """Check whether the current measurement is clipping.

        Two independent checks are performed:

        1. **Status register** — the hardware overload flag.
        2. **Trace proximity** — whether the peak trace amplitude is within
           *margin_db* of the reference level, which typically indicates the
           ADC is saturating even if the status bit is not set.

        Returns a dict::

            {
                "clipping": bool,           # True if either check triggers
                "status_overload": bool,    # hardware overload flag
                "trace_near_ref": bool,     # peak within margin of ref level
                "ref_level_dbm": float,
                "peak_dbm": float,
                "input_atten_db": float,
            }
        """
        status_overload = self.is_input_overloaded()
        ref_level = self.get_reference_level()
        atten = self.get_input_attenuation()
        _, amps = self.read_trace(trace)
        peak_dbm = float(np.max(amps))
        trace_near_ref = (ref_level - peak_dbm) < margin_db

        return {
            "clipping": status_overload or trace_near_ref,
            "status_overload": status_overload,
            "trace_near_ref": trace_near_ref,
            "ref_level_dbm": ref_level,
            "peak_dbm": peak_dbm,
            "input_atten_db": atten,
        }

    def auto_adjust_for_overload(
        self,
        ref_level_step_db: float = 10.0,
        max_retries: int = 3,
        margin_db: float = 1.0,
        settle_s: float = 0.15,
    ) -> dict:
        """Raise the reference level until clipping stops.

        Uses auto-attenuation so the SA couples the attenuator to the
        reference level.  If the SA still reports overload or the trace
        peak is within *margin_db* of the reference level, the reference
        level is raised by *ref_level_step_db* and the sweep is repeated.

        Returns the same dict as :meth:`check_clipping`, augmented with
        ``"converged"`` (``True`` if clipping was resolved).
        """
        for attempt in range(max_retries + 1):
            time.sleep(settle_s)
            self.trigger_single_sweep()
            status = self.check_clipping(margin_db=margin_db)
            if not status["clipping"]:
                return {**status, "converged": True}
            current_ref = status["ref_level_dbm"]
            new_ref = current_ref + ref_level_step_db
            log.warning(
                "SA overload (attempt %d/%d). "
                "Raising reference level %.1f -> %.1f dBm.",
                attempt + 1, max_retries, current_ref, new_ref,
            )
            self._apply_ref_auto(new_ref)

        return {**status, "converged": False}

    # -- Sweep control ---------------------------------------------------------

    def set_continuous_sweep(self, enabled: bool) -> None:
        self.write(f":INITiate:CONTinuous {'ON' if enabled else 'OFF'}")

    def get_sweep_time(self) -> float:
        """Return the current sweep time in seconds."""
        return float(self.query(":SENSe:SWEep:TIME?"))

    def trigger_single_sweep(self, timeout_s: float = 30.0) -> None:
        """Start a single sweep and block until it completes.

        Rigol SAs do not honour ``*OPC?`` for sweep synchronisation, so
        we query the expected sweep time and sleep for that duration plus
        a safety margin, then confirm via ``*OPC?`` as a belt-and-suspenders.
        """
        self.write(":TRACe1:MODE WRITe")
        self.write(":INITiate:CONTinuous OFF")
        self.write(":ABORt")
        time.sleep(0.1)

        sweep_time_s = self.get_sweep_time()

        self.write("*CLS")
        self.write(":INITiate:IMMediate")

        time.sleep(sweep_time_s * 1.2 + 0.5)

        try:
            self._opc_wait(timeout_s=timeout_s)
        except Exception:
            time.sleep(0.5)

    def set_sweep_points(self, n_points: int) -> None:
        self.write(f":SENSe:SWEep:POINts {n_points}")

    def get_sweep_points(self) -> int:
        return int(float(self.query(":SENSe:SWEep:POINts?")))

    def set_sweep_time(self, time_s: float) -> None:
        self.write(":SENSe:SWEep:TIME:AUTO OFF")
        self.write(f":SENSe:SWEep:TIME {time_s:.6f}")

    def set_sweep_time_auto(self) -> None:
        self.write(":SENSe:SWEep:TIME:AUTO ON")

    # -- Trace control ---------------------------------------------------------

    def set_trace_mode(self, trace: int = 1, mode: str = "WRITe") -> None:
        """Set trace mode: ``WRITe``, ``MAXHold``, ``MINHold``, ``VIEW``, ``BLANk``, ``AVERage``."""
        self.write(f":TRACe{trace}:MODE {mode}")

    def set_trace_average_count(self, count: int) -> None:
        self.write(f":SENSe:AVERage:COUNt {count}")

    # -- Marker operations -----------------------------------------------------

    def set_marker_state(self, marker: int = 1, enabled: bool = True) -> None:
        self.write(f":CALCulate:MARKer{marker}:STATe {'ON' if enabled else 'OFF'}")

    def set_marker_frequency(self, freq_hz: float, marker: int = 1) -> None:
        self.set_marker_state(marker, True)
        self.write(f":CALCulate:MARKer{marker}:X {freq_hz:.0f}")

    def marker_peak_search(self, marker: int = 1) -> None:
        self.set_marker_state(marker, True)
        self.write(f":CALCulate:MARKer{marker}:MAXimum")

    def marker_next_peak(self, marker: int = 1) -> None:
        self.write(f":CALCulate:MARKer{marker}:MAXimum:NEXT")

    def read_marker(self, marker: int = 1) -> Tuple[float, float]:
        freq_hz = float(self.query(f":CALCulate:MARKer{marker}:X?"))
        amp_dbm = float(self.query(f":CALCulate:MARKer{marker}:Y?"))
        return freq_hz, amp_dbm

    # -- Trace data readout ----------------------------------------------------

    def read_trace(self, trace: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        start = float(self.query(":SENSe:FREQuency:STARt?"))
        stop = float(self.query(":SENSe:FREQuency:STOP?"))

        raw = self.query(f":TRACe:DATA? TRACE{trace}")
        amplitudes = np.array([float(x) for x in raw.split(",")], dtype=np.float64)
        frequencies = np.linspace(start, stop, len(amplitudes))
        return frequencies, amplitudes

    def read_trace_fresh(
        self,
        trace: int = 1,
        max_retries: int = 3,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Trigger a sweep and read trace, retrying if data looks stale.

        Compares an MD5 hash of the amplitude array against the previous
        read.  If identical (sweep didn't update the buffer), re-triggers
        up to *max_retries* times with progressively longer waits.
        """
        for attempt in range(max_retries):
            self.trigger_single_sweep()
            freqs, amps = self.read_trace(trace)

            data_hash = hashlib.md5(amps.tobytes()).digest()
            if data_hash != self._last_trace_hash:
                self._last_trace_hash = data_hash
                return freqs, amps

            extra_wait = 0.5 * (attempt + 1)
            time.sleep(extra_wait)

        self._last_trace_hash = data_hash
        return freqs, amps

    # -- Convenience -----------------------------------------------------------

    def configure_for_single_tone(
        self,
        center_hz: float,
        span_hz: float = 1e6,
        rbw_hz: Optional[float] = None,
        vbw_hz: Optional[float] = None,
        ref_level_dbm: float = 10.0,
    ) -> None:
        """One-shot setup for measuring a single tone at *center_hz*."""
        self._set_and_verify(
            f":SENSe:FREQuency:CENTer {center_hz:.0f}",
            ":SENSe:FREQuency:CENTer?",
            center_hz,
        )
        self._set_and_verify(
            f":SENSe:FREQuency:SPAN {span_hz:.0f}",
            ":SENSe:FREQuency:SPAN?",
            span_hz,
        )
        if rbw_hz is not None:
            self.set_rbw(rbw_hz)
        else:
            self.set_rbw_auto()
        if vbw_hz is not None:
            self.set_vbw(vbw_hz)
        else:
            self.set_vbw_auto()
        self._apply_ref_auto(ref_level_dbm)

    def measure_power_at_frequency(
        self,
        freq_hz: float,
        span_hz: float = 1e6,
        rbw_hz: Optional[float] = None,
        ref_level_dbm: float = 10.0,
        settle_s: float = 0.1,
        max_overload_retries: int = 3,
        ref_level_step_db: float = 10.0,
    ) -> Tuple[float, float]:
        """Measure the peak power near *freq_hz*.

        Configures center/span, runs a single sweep, does a peak search,
        and returns ``(measured_freq_hz, power_dbm)``.

        Auto-attenuation is enabled and given time to settle before each
        sweep.  If the SA reports overload after a sweep the data is
        discarded, the reference level is raised by *ref_level_step_db*
        (so auto-attenuation increases to match), and the sweep is
        repeated.  Marker/trace data is only read once a clean sweep
        has been confirmed.
        """
        self.configure_for_single_tone(
            center_hz=freq_hz,
            span_hz=span_hz,
            rbw_hz=rbw_hz,
            ref_level_dbm=ref_level_dbm,
        )
        if settle_s > 0:
            time.sleep(settle_s)

        current_ref = ref_level_dbm
        for attempt in range(max_overload_retries + 1):
            self.trigger_single_sweep()
            if not self.is_input_overloaded():
                break
            new_ref = current_ref + ref_level_step_db
            log.warning(
                "SA input overload detected (attempt %d/%d). "
                "Raising reference level %.1f -> %.1f dBm and re-sweeping.",
                attempt + 1, max_overload_retries, current_ref, new_ref,
            )
            current_ref = new_ref
            self._apply_ref_auto(current_ref)
            if settle_s > 0:
                time.sleep(settle_s)

        self.marker_peak_search(marker=1)
        return self.read_marker(marker=1)

    def measure_harmonics(
        self,
        fundamental_hz: float,
        n_harmonics: int = 5,
        per_tone_span_hz: float = 1e6,
        rbw_hz: Optional[float] = None,
        ref_level_dbm: float = 10.0,
        settle_s: float = 0.1,
        max_overload_retries: int = 3,
        ref_level_step_db: float = 10.0,
    ) -> dict:
        """Measure the fundamental and its harmonics from a wideband sweep.

        Takes a wideband sweep covering all harmonics, then extracts the
        peak power near each expected harmonic frequency from the trace
        data.

        Auto-attenuation is enabled and given time to settle before each
        sweep.  After every sweep the overload status register is checked
        **before** the trace data is read.  If overload occurred the
        sweep data is discarded, the reference level is raised by
        *ref_level_step_db* (so auto-attenuation increases), and the
        sweep is repeated.  Trace data is only read once a clean sweep
        has been confirmed.

        Parameters
        ----------
        fundamental_hz : float
            Fundamental frequency.
        n_harmonics : int
            Number of harmonics to measure (including fundamental).
            E.g. 5 → measures f, 2f, 3f, 4f, 5f.
        per_tone_span_hz : float
            Window around each nominal harmonic within which the peak
            is searched in the wideband trace data.
        rbw_hz : float, optional
            Resolution bandwidth.  ``None`` → auto.
        ref_level_dbm : float
            Reference level for all measurements.
        settle_s : float
            Settle time after configuring the SA before sweeping.
        max_overload_retries : int
            Max times to raise reference level and re-sweep on overload.
        ref_level_step_db : float
            How much to raise the reference level on each overload retry.

        Returns
        -------
        dict with keys:

        - ``"fundamental_hz"`` : float
        - ``"harmonics"`` : list of dicts, each with
          ``harmonic_number``, ``nominal_freq_hz``,
          ``measured_freq_hz``, ``power_dbm``
        - ``"wideband_trace"`` : ``(frequencies_hz, amplitudes_dbm)``
          covering 0.5f … (n_harmonics + 0.5) × f0
        """
        wideband_start = max(fundamental_hz * 0.5, 9e3)
        wideband_stop = min((n_harmonics + 0.5) * fundamental_hz, 3.0e9)

        self._set_and_verify(
            f":SENSe:FREQuency:STARt {wideband_start:.0f}",
            ":SENSe:FREQuency:STARt?",
            wideband_start,
        )
        self._set_and_verify(
            f":SENSe:FREQuency:STOP {wideband_stop:.0f}",
            ":SENSe:FREQuency:STOP?",
            wideband_stop,
        )

        span = wideband_stop - wideband_start
        min_points = max(int(span / (per_tone_span_hz * 0.25)) + 1, 801)
        min_points = min(min_points, 10001)
        self._set_and_verify(
            f":SENSe:SWEep:POINts {min_points}",
            ":SENSe:SWEep:POINts?",
            min_points,
        )
        self.set_sweep_time_auto()

        if rbw_hz is not None:
            self.set_rbw(rbw_hz)
        else:
            self.set_rbw_auto()
        self.set_vbw_auto()

        current_ref = ref_level_dbm
        self._apply_ref_auto(current_ref)
        time.sleep(max(settle_s, 0.1))

        # -- Sweep with overload retry (sweep first, check, then read) ---------
        for attempt in range(max_overload_retries + 1):
            self.trigger_single_sweep()

            if not self.is_input_overloaded():
                break

            new_ref = current_ref + ref_level_step_db
            log.warning(
                "SA input overload during harmonic sweep (attempt %d/%d). "
                "Raising reference level %.1f -> %.1f dBm and re-sweeping.",
                attempt + 1, max_overload_retries, current_ref, new_ref,
            )
            current_ref = new_ref
            self._apply_ref_auto(current_ref)
            if settle_s > 0:
                time.sleep(settle_s)

        wb_freqs, wb_amps = self.read_trace(trace=1)

        # -- Extract harmonics from the clean trace ----------------------------
        point_spacing = span / max(len(wb_amps) - 1, 1)
        half_search = max(per_tone_span_hz / 2.0, point_spacing * 2.0)
        results = []
        for k in range(1, n_harmonics + 1):
            tone_hz = k * fundamental_hz
            if tone_hz > 3.0e9:
                break

            mask = (wb_freqs >= tone_hz - half_search) & (wb_freqs <= tone_hz + half_search)
            if np.any(mask):
                idx = np.argmax(wb_amps[mask])
                peak_freq = float(wb_freqs[mask][idx])
                peak_power = float(wb_amps[mask][idx])
            else:
                peak_freq = tone_hz
                peak_power = float("nan")

            results.append({
                "harmonic_number": k,
                "nominal_freq_hz": tone_hz,
                "measured_freq_hz": peak_freq,
                "power_dbm": peak_power,
            })

        return {
            "fundamental_hz": fundamental_hz,
            "harmonics": results,
            "wideband_trace": (wb_freqs, wb_amps),
        }
