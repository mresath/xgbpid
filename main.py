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

    live_path     = output_dir.parent / "live_telemetry.parquet"
    flush_interval = int(cfg["dashboard"]["flush_interval"])
    rolling_window = int(cfg["dashboard"]["rolling_window"])

    daq = build_daq(cfg)
    clf = PIDClassifier.from_config(cfg)
    proc = cfg["processing"]

    # accumulate rows in plain lists; convert to Polars once at the end
    rows: list[dict] = []

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
                    **{k: float(v) for k, v in zip(FeatureVector.feature_names(), fv.to_array())},
                })

                # flush after latency measurement — outside the < 1 ms hot path
                if event_id % flush_interval == 0:
                    _write_live_telemetry(rows, live_path, rolling_window)

                event_id += 1

        except CriticalDAQError:
            log.critical("Unrecoverable hardware failure. Stopping acquisition.")
        except KeyboardInterrupt:
            log.info("Interrupted after %d events (%d pile-up skipped).", event_id, pileup_ct)
        finally:
            _write_live_telemetry(rows, live_path, rolling_window)  # flush remainder

    if rows:
        run_ts  = time.strftime("%Y%m%d_%H%M%S")
        outfile = output_dir / f"run_{run_ts}.parquet"
        pl.DataFrame(rows).write_parquet(outfile)
        log.info("Telemetry written to '%s' (%d events).", outfile, len(rows))
    else:
        log.warning("No events recorded; nothing written.")


def main() -> None:
    parser = argparse.ArgumentParser(description="BL4S AI PID — T9 beamline acquisition")
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