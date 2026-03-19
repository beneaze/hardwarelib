# hardwarelib

General-purpose Python driver library for lab hardware.

## Structure

Drivers are organised by **instrument type / manufacturer / model**:

```
hardwarelib/
├── base.py                          # Abstract base classes (Oscilloscope, SignalGenerator, …)
├── oscilloscopes/
│   └── rigol/
│       └── dho4000.py               # Rigol DHO/HDO 4000 series
└── signal_generators/
    └── windfreak/
        └── synthhd.py               # Windfreak SynthHD dual-channel RF synthesizer
```

## Installation

```bash
pip install -e .
```

## Quick start

```python
from hardwarelib.oscilloscopes.rigol import RigolDHO4000
from hardwarelib.signal_generators.windfreak import WindfreakSynthHD

with RigolDHO4000("TCPIP0::192.168.1.50::INSTR") as scope:
    print(scope.idn())
    scope.acquire_single_and_wait(timeout_s=4.0)
    t, v, meta = scope.read_waveform(channel=1)

with WindfreakSynthHD("COM12", channel=0) as rf:
    rf.apply(freq_hz=1e9, power_dbm=-10.0, enabled=True)
```

## Adding a new instrument

1. Pick the correct category folder under `hardwarelib/` (e.g. `oscilloscopes/`, `signal_generators/`, `power_meters/`, …). Create it if it doesn't exist.
2. Create a sub-folder for the manufacturer.
3. Add a Python module for the specific model.
4. Subclass the appropriate base class from `hardwarelib.base` and implement the abstract methods.
