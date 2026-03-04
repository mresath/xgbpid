"""
Feature extraction for raw scintillator pulses.

Takes an AcquisitionBuffer (1024 float32 samples in volts) and returns a
FeatureVector ready to be handed to the classifier. All timing features are
in nanoseconds; charge features are in V·sample (proportional to ADC counts).
"""

import logging
from dataclasses import dataclass

import numpy as np

from xgbpid.core.daq import AcquisitionBuffer

log = logging.getLogger(__name__)

# 125 MHz at decimation=1 → 8 ns per sample
NS_PER_SAMPLE: float = 8.0


@dataclass
class FeatureVector:
    """Physics-derived features extracted from one triggered pulse."""
    rise_time_ns: float        # 10%–90% amplitude crossing time
    fwhm_ns: float             # full width at half maximum
    tail_to_total: float       # PSD ratio: tail integral / total integral
    auc: float                 # total area under curve (V·sample)
    peak_amplitude: float      # max sample value (V)
    baseline_rms: float        # RMS noise of the pre-trigger window (V)

    def to_array(self) -> np.ndarray:
        """Return features as a 1-D float32 array in a fixed, stable order."""
        return np.array(
            [
                self.rise_time_ns,
                self.fwhm_ns,
                self.tail_to_total,
                self.auc,
                self.peak_amplitude,
                self.baseline_rms,
            ],
            dtype=np.float32,
        )

    @classmethod
    def feature_names(cls) -> list[str]:
        return ["rise_time_ns", "fwhm_ns", "tail_to_total", "auc", "peak_amplitude", "baseline_rms"]


def _count_peaks_above(samples: np.ndarray, threshold: float, min_width: int = 3) -> int:
    """
    Count distinct positive excursions above threshold that span at least
    min_width consecutive samples. Single-sample noise transients are ignored.
    """
    above = samples > threshold
    count  = 0
    run    = 0
    in_peak = False
    for v in above:
        if v:
            run += 1
            if run >= min_width and not in_peak:
                count  += 1
                in_peak = True
        else:
            run     = 0
            in_peak = False
    return count


def _find_threshold_crossing(
    samples: np.ndarray,
    threshold: float,
    peak_idx: int,
    direction: str,
) -> float:
    """
    Return the interpolated sample index where the waveform crosses `threshold`.

    direction='rising'  — searches left of peak_idx
    direction='falling' — searches right of peak_idx
    Returns peak_idx (as float) if no crossing is found.
    """
    if direction == "rising":
        indices = np.where(samples[:peak_idx] < threshold)[0]
        if len(indices) == 0:
            return float(peak_idx)
        i = indices[-1]
        # linear interpolation between sample i and i+1
        dy = samples[i + 1] - samples[i]
        if dy == 0:
            return float(i)
        return i + (threshold - samples[i]) / dy

    else:  # falling
        indices = np.where(samples[peak_idx:] < threshold)[0]
        if len(indices) == 0:
            return float(len(samples) - 1)
        i = peak_idx + indices[0]
        if i == 0:
            return float(i)
        dy = samples[i] - samples[i - 1]
        if dy == 0:
            return float(i)
        return (i - 1) + (threshold - samples[i - 1]) / dy


def extract(
    buf: AcquisitionBuffer,
    tail_start_frac: float = 0.25,
    pileup_threshold_v: float = 0.050,
    pileup_min_width: int = 5,
    rise_low_frac: float = 0.10,
    rise_high_frac: float = 0.90,
    baseline_window: int = 50,
) -> "FeatureVector | None":
    """
    Compute the full feature vector from a raw acquisition buffer.

    Returns None if the event is flagged as pile-up (more than one distinct
    pulse above pileup_threshold_v), so the caller can discard it cleanly.

    pileup_min_width sets how many consecutive samples must stay above
    pileup_threshold_v before a crossing counts as a real second pulse.
    The default of 5 samples (40 ns at 125 MHz) prevents long-tail species
    such as kaons from being falsely rejected when noise briefly dips the
    waveform below the threshold during the decay.

    All threshold parameters are read from the experiment config so the system
    can be recalibrated for the CERN noise floor without a code redeploy.
    """
    wf = buf.samples.astype(np.float64)
    n  = len(wf)

    # baseline from the configured pre-trigger window
    baseline     = float(np.mean(wf[:baseline_window]))
    baseline_rms = float(np.std(wf[:baseline_window]))
    wf -= baseline   # zero-centre

    # pile-up rejection: more than one real pulse (min_width-sample minimum width)
    if _count_peaks_above(wf, pileup_threshold_v, min_width=pileup_min_width) > 1:
        log.debug("Pile-up detected; discarding event.")
        return None

    peak_idx = int(np.argmax(wf))
    peak_amp = float(wf[peak_idx])

    if peak_amp <= 0:
        log.warning("Non-positive peak amplitude (%.4f V); returning zero features.", peak_amp)
        return FeatureVector(0.0, 0.0, 0.0, 0.0, 0.0, baseline_rms)

    # rise time using configurable low/high fractions of peak
    t10 = _find_threshold_crossing(wf, rise_low_frac * peak_amp, peak_idx, "rising")
    t90 = _find_threshold_crossing(wf, rise_high_frac * peak_amp, peak_idx, "rising")
    rise_time_ns = (t90 - t10) * NS_PER_SAMPLE

    # FWHM
    half_max = 0.50 * peak_amp
    t_left   = _find_threshold_crossing(wf, half_max, peak_idx, "rising")
    t_right  = _find_threshold_crossing(wf, half_max, peak_idx, "falling")
    fwhm_ns  = (t_right - t_left) * NS_PER_SAMPLE

    # area under curve (trapezoidal, clipped to positive values)
    wf_pos = np.maximum(wf, 0.0)
    auc    = float(np.trapezoid(wf_pos))

    # tail-to-total PSD ratio
    tail_start = int(n * tail_start_frac)
    tail_auc   = float(np.trapezoid(wf_pos[tail_start:]))
    tail_to_total = tail_auc / auc if auc > 0 else 0.0

    return FeatureVector(
        rise_time_ns=rise_time_ns,
        fwhm_ns=fwhm_ns,
        tail_to_total=tail_to_total,
        auc=auc,
        peak_amplitude=float(peak_amp),
        baseline_rms=baseline_rms,
    )
