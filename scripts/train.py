"""
Model training script.

Generates synchronized (waveform, teacher_label) pairs via MockDAQ — which
mirrors the Teacher-Student setup on the real beamline — extracts physics
features, and trains a three-class XGBoost classifier.

Label convention:
    0 = pion      (XCET-1 fired, no calorimeter)
    1 = electron  (XCET-1 fired, calorimeter above threshold)
    2 = kaon      (XCET-2 fired, XCET-1 vetoed)

The model is trained with balanced class sizes by default so the classifier
learns discriminating features for each species equally.  At inference time
the real beam fractions (~8% e⁻, ~1.5% K⁻, rest π) are naturally reflected
in the event-by-event population seen by main.py.

Usage:
    python -m scripts.train
    python -m scripts.train --n-events 30000 --output models/xgbpid.json
"""

import argparse
import logging
from pathlib import Path

import polars as pl
import xgboost as xgb
import yaml

from xgbpid.core.daq import MockDAQ
from xgbpid.core.processor import FeatureVector, extract

log = logging.getLogger(__name__)


def _generate_dataset(cfg: dict, n_events: int) -> pl.DataFrame:
    """
    Run MockDAQ for each particle species separately, extract features, and
    assemble a Polars DataFrame with balanced class sizes. Training on equal class counts prevents the classifier from learning the
    beam composition rather than the pulse-shape differences.

    Events flagged as pile-up by extract() are dropped; the final
    dataset may therefore be slightly smaller than n_events.
    """
    hw_cfg  = cfg["hardware"]
    sim_cfg = cfg["simulation"]

    rows: list[dict] = []
    n_per_class = n_events // 3
    proc = cfg.get("processing", {})

    for label, teacher_label in (("electron", 1), ("pion", 0), ("kaon", 2)):
        daq = MockDAQ(
            buffer_size=hw_cfg["buffer_size"],
            noise_sigma_frac=sim_cfg["noise_sigma_frac"],
            pulse_config=cfg["pulse_shapes"],
            label=label,
        )
        discarded = 0
        with daq:
            for _ in range(n_per_class):
                buf = daq.wait_for_trigger()
                fv  = extract(
                    buf,
                    pileup_threshold_v=proc.get("pileup_threshold_v", 0.050),
                    pileup_min_width=proc.get("pileup_min_width", 5),
                    rise_low_frac=proc.get("rise_low_frac", 0.10),
                    rise_high_frac=proc.get("rise_high_frac", 0.90),
                    baseline_window=proc.get("baseline_window", 50),
                )
                if fv is None:
                    discarded += 1
                    continue
                row = dict(zip(FeatureVector.feature_names(), fv.to_array().tolist()))
                row["teacher_label"] = teacher_label
                rows.append(row)

        if discarded:
            log.warning("%s: %d events discarded as pile-up.", label, discarded)

    log.info("Dataset ready: %d events (requested %d, 3 classes).", len(rows), n_events)
    return pl.DataFrame(rows)


def train(cfg: dict, n_events: int, output_path: Path) -> None:
    df = _generate_dataset(cfg, n_events)

    feature_cols = FeatureVector.feature_names()
    X = df.select(feature_cols).to_numpy()
    y = df["teacher_label"].to_numpy()

    dtrain = xgb.DMatrix(X, label=y, feature_names=feature_cols)

    params: dict = {
        "objective":        "multi:softprob",
        "num_class":        3,
        "eval_metric":      "mlogloss",
        "max_depth":        4,
        "eta":              0.1,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "seed":             42,
    }

    log.info("Training XGBoost (100 rounds, 3-class) …")
    booster = xgb.train(params, dtrain, num_boost_round=100, verbose_eval=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(output_path))
    log.info("Model saved to '%s'.", output_path)

    # quick feature importance summary
    scores = booster.get_score(importance_type="gain")
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    log.info("Feature importance (gain): %s", "  ".join(f"{k}={v:.1f}" for k, v in ranked))


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser(description="Train the XGBPID classifier")
    parser.add_argument("--config",   default="configs/experiment_v1.yaml")
    parser.add_argument("--n-events", type=int, default=30_000, dest="n_events")
    parser.add_argument("--output",   default="models/xgbpid.json")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    train(cfg, args.n_events, Path(args.output))


if __name__ == "__main__":
    main()