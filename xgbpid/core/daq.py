"""
Data acquisition.

Two classes live here:
  - RedPitayaDAQ  : talks to real hardware over SCPI/TCP.
  - MockDAQ       : generates synthetic scintillator pulses for offline dev.

Which class is used at runtime is decided by `use_simulation` in the config.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np

from xgbpid.core.redpitaya_scpi import scpi

log = logging.getLogger(__name__)


class CriticalDAQError(RuntimeError):
    """Raised when the Red Pitaya cannot be recovered after a hardware reset."""


@dataclass
class AcquisitionBuffer:
    """One event's worth of raw ADC samples."""
    samples: np.ndarray          # shape (buffer_size,), float32 in volts
    timestamp_ns: float          # wall-clock time of trigger receipt
    channel: str                 # e.g. "CH1"
    teacher_label: int | None = None  # 1 = electron (Cherenkov fired), 0 = pion, None = unknown
    simulated: bool = False


class BaseDAQ(ABC):
    """Common interface for data acquisition."""

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def wait_for_trigger(self) -> AcquisitionBuffer:
        """Block until the next trigger fires, then return the raw buffer."""
        ...

    def __enter__(self) -> "BaseDAQ":
        self.connect()
        return self

    def __exit__(self, *_) -> None:
        self.disconnect()


class RedPitayaDAQ(BaseDAQ):
    """
    Drives a Red Pitaya board via SCPI over TCP.

    The acquisition loop follows the TLU state machine:
      WAIT  →  (beam arrives)  →  TD  →  pull buffer  →  reset  →  WAIT

    Ref: Red Pitaya SCPI command set, ACQ section.
    """

    _TRIGGER_POLL_S: float = 1e-3   # 1 ms sleep between polls — prevents CPU spin during beam gaps

    def __init__(
        self,
        host: str,
        port: int,
        channel: str,
        decimation: int,
        buffer_size: int,
        trigger_level: float,
        trigger_delay: int,
    ) -> None:
        self._host = host
        self._port = port
        self._channel = channel
        self._decimation = decimation
        self._buffer_size = buffer_size
        self._trigger_level = trigger_level
        self._trigger_delay = trigger_delay
        self._rp: scpi | None = None

    def connect(self) -> None:
        log.info("Connecting to Red Pitaya at %s:%d …", self._host, self._port)
        try:
            self._rp = scpi(self._host, self._port)
            self._configure()
            log.info("Red Pitaya connected.")
        except OSError as exc:
            log.critical("Could not reach Red Pitaya: %s", exc)
            raise

    def disconnect(self) -> None:
        if self._rp is not None:
            self._rp.close()
            self._rp = None
            log.info("Red Pitaya disconnected.")

    def _configure(self) -> None:
        """Push acquisition settings to the board."""
        rp = self._rp
        assert rp is not None

        rp.tx_txt("ACQ:RST")
        rp.tx_txt(f"ACQ:DEC {self._decimation}")
        rp.tx_txt(f"ACQ:TRIG:LEVEL {self._trigger_level}")
        rp.tx_txt(f"ACQ:TRIG:DELAY {self._trigger_delay}")
        rp.tx_txt(f"ACQ:{self._channel}:GAIN LV")
        log.debug("ACQ configured: DEC=%d  TRIG_LVL=%.3f V", self._decimation, self._trigger_level)

    def wait_for_trigger(self) -> AcquisitionBuffer:
        """
        Arms the oscilloscope, blocks until the TLU fires (state == 'TD'),
        then pulls the full 1024-sample buffer from CH1.

        On connection failure: logs critical, attempts one reconnect,
        then re-raises to let the caller decide whether to abort.
        """
        rp = self._rp
        assert rp is not None, "Call connect() before acquiring."

        try:
            rp.tx_txt("ACQ:START")
            rp.tx_txt("ACQ:TRIG EXT_PE")   # arm for external positive edge

            # poll until TLU fires; sleep prevents CPU spin during beam gaps
            while True:
                rp.tx_txt("ACQ:TRIG:STAT?")
                state = rp.rx_txt().strip()
                if state == "TD":
                    break
                time.sleep(self._TRIGGER_POLL_S)

            t_trigger = float(time.time_ns())  # wall-clock ns — syncs with CERN run logs

            log.debug("Trigger received at t=%.0f ns", t_trigger)

            # CH1 — student waveform (scintillator)
            rp.tx_txt(f"ACQ:SOUR{self._channel[-1]}:DATA?")
            raw_str = rp.rx_txt().strip().strip("{}")
            samples = np.fromstring(raw_str, sep=",", dtype=np.float32)

            # CH2 — teacher label (Cherenkov XCET: fires on electrons)
            rp.tx_txt("ACQ:SOUR2:DATA?")
            ch2_str = rp.rx_txt().strip().strip("{}")
            ch2 = np.fromstring(ch2_str, sep=",", dtype=np.float32)
            teacher_label = 1 if float(ch2.max()) > self._trigger_level else 0

            return AcquisitionBuffer(
                samples=samples,
                timestamp_ns=t_trigger,
                channel=self._channel,
                teacher_label=teacher_label,
                simulated=False,
            )

        except OSError as exc:
            log.critical("Lost connection: %s — attempting hardware reset.", exc)
            self.disconnect()
            try:
                self.connect()  # _configure() issues ACQ:RST as its first command
            except OSError as reset_exc:
                raise CriticalDAQError(
                    "Hardware reset failed; cannot recover Red Pitaya."
                ) from reset_exc
            raise RuntimeError("Trigger missed due to reconnection.") from exc


ParticleLabel = Literal["electron", "pion"]

class MockDAQ(BaseDAQ):
    """
    Simulated stand-in for RedPitayaDAQ.

    Generates synthetic scintillator pulses that loosely match what a real
    PMT + scintillator detector produces in the CERN T9 beamline.

    Pulse model (both species):
        V(t) = A · [ exp(-t/τ_rise) - exp(-t/τ_fall) ]  +  ε(t)

    where ε ~ N(0, σ²) with σ = noise_sigma_frac · A.

    Shape parameters differ by species to give the classifier something to
    learn during the training phase.
    """

    # sample rate at DEC=1
    _SAMPLE_RATE_HZ: float = 125e6          # 125 MHz
    _NS_PER_SAMPLE:  float = 1e9 / _SAMPLE_RATE_HZ   # 8 ns

    def __init__(
        self,
        buffer_size: int,
        noise_sigma_frac: float,
        pulse_config: dict,
        label: ParticleLabel | None = None,
        electron_fraction: float = 0.08,
        onset_jitter_samples: int = 5,
    ) -> None:
        self._buffer_size = buffer_size
        self._noise_sigma_frac = noise_sigma_frac
        self._pulse_config = pulse_config
        self._label = label                          # None → sample from beam composition
        self._electron_fraction = electron_fraction
        self._onset_jitter = onset_jitter_samples
        self._rng = np.random.default_rng()

    def connect(self) -> None:
        if self._label is None:
            log.info(
                "MockDAQ ready (beam mixture: %.0f%% e⁻ / %.0f%% π, N=%d samples).",
                self._electron_fraction * 100,
                (1 - self._electron_fraction) * 100,
                self._buffer_size,
            )
        else:
            log.info("MockDAQ ready (label=%s, N=%d samples).", self._label, self._buffer_size)

    def disconnect(self) -> None:
        log.info("MockDAQ closed.")

    def set_label(self, label: ParticleLabel) -> None:
        """Switch the particle type that will be simulated on the next call."""
        self._label = label

    def wait_for_trigger(self) -> AcquisitionBuffer:
        """
        Returns a synthetic pulse instantly (no actual blocking).
        The pulse onset is jittered ±onset_jitter_samples around sample 50,
        matching the timing uncertainty of a real external trigger.

        When label is None (beam-mixture mode), species is sampled per-event
        using electron_fraction, matching real beam composition.
        Amplitude is drawn from a species-dependent distribution:
          - Electron: Gaussian (EM shower — tighter spread)
          - Pion:     LogNormal (Landau/MIP — right-tailed)
        """
        # species selection
        if self._label is None:
            species: ParticleLabel = (
                "electron" if self._rng.random() < self._electron_fraction else "pion"
            )
        else:
            species = self._label

        cfg       = self._pulse_config[species]
        A_nominal = float(cfg["amplitude_v"])
        spread    = float(cfg.get("amplitude_spread_frac", 0.10))

        # amplitude variation by species
        if species == "pion":
            # LogNormal with the given fractional spread gives a Landau-like right tail
            sigma_ln = float(np.sqrt(np.log(1.0 + spread ** 2)))
            mu_ln    = float(np.log(A_nominal) - 0.5 * sigma_ln ** 2)
            A = float(np.clip(self._rng.lognormal(mu_ln, sigma_ln), 0.05, None))
        else:
            A = float(np.clip(self._rng.normal(A_nominal, spread * A_nominal), 0.05, None))

        τ_rise = float(cfg["rise_ns"])  / self._NS_PER_SAMPLE
        τ_fall = float(cfg["decay_ns"]) / self._NS_PER_SAMPLE

        t = np.arange(self._buffer_size, dtype=np.float32)

        # timing jitter: uniform ±onset_jitter_samples around sample 50
        onset = int(np.clip(
            50 + self._rng.integers(-self._onset_jitter, self._onset_jitter + 1),
            10, self._buffer_size - 100,
        ))
        t_rel = np.maximum(t - onset, 0.0)

        pulse = A * (np.exp(-t_rel / τ_fall) - np.exp(-t_rel / τ_rise))
        pulse = np.clip(pulse, 0.0, None)

        sigma = A * self._noise_sigma_frac
        noise = self._rng.normal(0.0, sigma, size=self._buffer_size).astype(np.float32)

        teacher_label = 1 if species == "electron" else 0

        return AcquisitionBuffer(
            samples=(pulse + noise).astype(np.float32),
            timestamp_ns=float(time.time_ns()),
            channel="SIM",
            teacher_label=teacher_label,
            simulated=True,
        )


def build_daq(cfg: dict) -> BaseDAQ:
    """
    Instantiate the right DAQ class from a parsed YAML config dict.

    >>> daq = build_daq(config["hardware"] | {"use_simulation": True, ...})
    """
    use_sim: bool = cfg["simulation"]["use_simulation"]

    if use_sim:
        log.info("Simulation mode — MockDAQ selected.")
        sim = cfg["simulation"]
        return MockDAQ(
            buffer_size=cfg["hardware"]["buffer_size"],
            noise_sigma_frac=sim["noise_sigma_frac"],
            pulse_config=cfg["pulse_shapes"],
            electron_fraction=sim.get("electron_fraction", 0.08),
            onset_jitter_samples=sim.get("onset_jitter_samples", 5),
        )

    log.info("Hardware mode — RedPitayaDAQ selected.")
    hw = cfg["hardware"]
    return RedPitayaDAQ(
        host=hw["host"],
        port=hw["port"],
        channel=hw["channel"],
        decimation=hw["decimation"],
        buffer_size=hw["buffer_size"],
        trigger_level=hw["trigger_level"],
        trigger_delay=hw["trigger_delay"],
    )
