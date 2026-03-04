"""
Main Acquisition Loop

Execution flow per beam spill:
    DAQ.wait_for_trigger()  →  processor.extract()  →  PIDClassifier.predict()
    →  log to Polars frame  →  repeat

Latency from trigger receipt to particle ID is measured every event and
logged at DEBUG level. Target: < 1 ms end-to-end.

Usage:
    python main.py [--config configs/experiment_v1.yaml]
"""

import argparse
import logging
import os
import time
from pathlib import Path

import polars as pl
import yaml

from xgbpid.core.daq import build_daq, CriticalDAQError
from xgbpid.core.inference import PIDClassifier
from xgbpid.core.processor import FeatureVector, extract
from xgbpid.core.relay import TelemetryRelay

log = logging.getLogger(__name__)


def _write_live_telemetry(rows: list[dict], path: Path, rolling: int) -> None:
    """Atomically overwrite *path* with the most recent *rolling* rows.

    Writes to a sibling .tmp file then calls os.replace() so the dashboard
    always reads a complete file — rename(2) is atomic on POSIX.
    """
    if not rows:
        return
    tmp = path.with_suffix(".parquet.tmp")
    pl.DataFrame(rows[-rolling:]).write_parquet(tmp)
    os.replace(tmp, path)          # atomic on POSIX
    log.debug("Live telemetry flushed → '%s' (%d rows).", path, min(len(rows), rolling))


def _write_kaon_telemetry(rows: list[dict], path: Path) -> None:
    """Atomically extend the kaon telemetry file with all buffered kaon rows.

    Kaons constitute only ~1-2% of the beam, so their events are written
    eagerly on every occurrence to prevent data loss between general flushes.
    """
    if not rows:
        return
    tmp = path.with_suffix(".parquet.tmp")
    pl.DataFrame(rows).write_parquet(tmp)
    os.replace(tmp, path)
    log.debug("Kaon telemetry flushed → '%s' (%d kaon events).", path, len(rows))


def _log_validation_metrics(
    rows: list[dict],
    window: int,
    target_electron_efficiency: float,
) -> None:
    recent = [r for r in rows[-window:] if r.get("is_correct") is not None]
    if not recent:
        return

    accuracy = sum(r["is_correct"] for r in recent) / len(recent)

    true_electrons = [r for r in recent if r["teacher_label"] == 1]
    true_pions     = [r for r in recent if r["teacher_label"] == 0]
    true_kaons     = [r for r in recent if r["teacher_label"] == 2]

    electron_eff = (
        sum(r["is_correct"] for r in true_electrons) / len(true_electrons)
        if true_electrons else float("nan")
    )
    pion_misid_rate = (
        sum(1 for r in true_pions if r["label"] == "electron") / len(true_pions)
        if true_pions else float("nan")
    )
    kaon_eff = (
        sum(r["is_correct"] for r in true_kaons) / len(true_kaons)
        if true_kaons else float("nan")
    )
    prf = 1.0 / pion_misid_rate if pion_misid_rate > 0 else float("inf")

    log.info(
        "Validation (n=%d): acc=%.3f  ε_e=%.3f  ε_K=%.3f  PRF=%.1f  (target ε_e=%.2f)",
        len(recent), accuracy, electron_eff, kaon_eff, prf, target_electron_efficiency,
    )


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        level=getattr(logging, level.upper(), logging.INFO),
    )


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run(cfg: dict) -> None:
    """
    Main loop.

    Runs until KeyboardInterrupt or until the hardware connection is
    unrecoverable. Telemetry is accumulated in memory and written to a
    Polars parquet file at the end of the run.
    """
    output_dir = Path(cfg["logging"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    live_path      = output_dir.parent / "live_telemetry.parquet"
    kaon_live_path = output_dir.parent / "kaon_telemetry.parquet"
    flush_interval = int(cfg["dashboard"]["flush_interval"])
    rolling_window = int(cfg["dashboard"]["rolling_window"])

    val_cfg = cfg.get("validation", {})
    validation_mode           = bool(val_cfg.get("enabled", False))
    validation_window         = int(val_cfg.get("window", 500))
    target_electron_efficiency = float(val_cfg.get("target_electron_efficiency", 0.90))

    if validation_mode:
        log.info(
            "Validation mode ON (window=%d, target ε_e=%.2f).",
            validation_window, target_electron_efficiency,
        )

    relay = TelemetryRelay(cfg, live_path)

    daq = build_daq(cfg)
    clf = PIDClassifier.from_config(cfg)
    proc = cfg["processing"]

    # accumulate rows in plain lists; convert to Polars once at the end
    rows: list[dict] = []
    kaon_rows: list[dict] = []  # dedicated buffer for kaon events — flushed on every occurrence

    log.info("Starting acquisition loop. Press Ctrl+C to stop.")
    event_id  = 0
    pileup_ct = 0

    with daq:
        try:
            while True:
                buf = daq.wait_for_trigger()
                t0  = time.perf_counter_ns()

                fv = extract(
                    buf,
                    pileup_threshold_v=proc["pileup_threshold_v"],
                    pileup_min_width=proc.get("pileup_min_width", 5),
                    rise_low_frac=proc["rise_low_frac"],
                    rise_high_frac=proc["rise_high_frac"],
                    baseline_window=proc["baseline_window"],
                )
                if fv is None:
                    pileup_ct += 1
                    log.debug("Pile-up event skipped (total skipped: %d).", pileup_ct)
                    continue

                pred = clf.predict(fv)

                latency_us = (time.perf_counter_ns() - t0) / 1e3   # µs

                is_correct: bool | None = None
                if validation_mode and buf.teacher_label is not None:
                    is_correct = pred.label_id == buf.teacher_label

                log.info(
                    "evt=%05d  %-8s  conf=%.3f  %s  latency=%.1f µs",
                    event_id,
                    pred.label,
                    pred.confidence,
                    "" if pred.above_threshold else "[LOW CONF]",
                    latency_us,
                )

                if latency_us > 1000:
                    log.warning("Latency budget exceeded: %.1f µs (target <1000 µs)", latency_us)

                rows.append({
                    "event_id":             event_id,
                    "nanosecond_timestamp": buf.timestamp_ns,
                    "teacher_label":        buf.teacher_label,
                    "label":           pred.label,
                    "label_id":        pred.label_id,
                    "confidence":      pred.confidence,
                    "above_threshold": pred.above_threshold,
                    "latency_us":      latency_us,
                    "is_correct":      is_correct,
                    **{k: float(v) for k, v in zip(FeatureVector.feature_names(), fv.to_array())},
                })

                # kaons are ~1–2% of the beam; flush immediately so low-rate events are never lost
                if pred.label_id == 2 or buf.teacher_label == 2:
                    kaon_rows.append(rows[-1])
                    _write_kaon_telemetry(kaon_rows, kaon_live_path)

                # flush and relay update — outside the < 1 ms hot path
                if event_id % flush_interval == 0:
                    _write_live_telemetry(rows, live_path, rolling_window)
                    relay.tick()
                    if validation_mode:
                        _log_validation_metrics(rows, validation_window, target_electron_efficiency)

                event_id += 1

        except CriticalDAQError:
            log.critical("Unrecoverable hardware failure. Stopping acquisition.")
        except KeyboardInterrupt:
            log.info("Interrupted after %d events (%d pile-up skipped).", event_id, pileup_ct)
        finally:
            _write_live_telemetry(rows, live_path, rolling_window)  # flush remainder
            _write_kaon_telemetry(kaon_rows, kaon_live_path)

    if rows:
        run_ts  = time.strftime("%Y%m%d_%H%M%S")
        outfile = output_dir / f"run_{run_ts}.parquet"
        pl.DataFrame(rows).write_parquet(outfile)
        log.info("Telemetry written to '%s' (%d events).", outfile, len(rows))
    else:
        log.warning("No events recorded; nothing written.")


def main() -> None:
    parser = argparse.ArgumentParser(description="XGBPID — T9 beamline acquisition")
    parser.add_argument(
        "--config",
        default="configs/experiment_v1.yaml",
        help="Path to experiment YAML config (default: configs/experiment_v1.yaml)",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    _setup_logging(cfg["logging"]["level"])
    run(cfg)


if __name__ == "__main__":
    main()