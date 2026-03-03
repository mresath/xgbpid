"""
XGBoost-based particle identification.

Loads a pre-trained model from disk and maps a FeatureVector to a
particle label (electron / pion) with an associated confidence score.
The model file is expected to be the JSON format produced by xgb.Booster.save_model().
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xgboost as xgb

from xgbpid.core.processor import FeatureVector

log = logging.getLogger(__name__)


@dataclass
class Prediction:
    """Result of one inference call."""
    label: str           # human-readable particle name, e.g. "electron"
    label_id: int        # integer class index
    confidence: float    # probability of the predicted class (0–1)
    above_threshold: bool  # True if confidence >= configured threshold


class PIDClassifier:
    """
    Wrapper around XGBoost Booster for particle ID.

    Usage:
        clf = PIDClassifier.from_config(cfg)
        pred = clf.predict(feature_vector)
    """

    def __init__(
        self,
        model_path: Path,
        labels: dict[int, str],
        confidence_threshold: float,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at '{model_path}'. "
                "Train and export a model first (see scripts/train.py)."
            )
        self._booster = xgb.Booster()
        self._booster.load_model(str(model_path))
        self._labels = labels
        self._threshold = confidence_threshold
        log.info("Loaded PID model from '%s' (threshold=%.2f).", model_path, confidence_threshold)

    @classmethod
    def from_config(cls, cfg: dict) -> "PIDClassifier":
        model_cfg = cfg["model"]
        return cls(
            model_path=Path(model_cfg["path"]),
            labels={int(k): v for k, v in model_cfg["labels"].items()},
            confidence_threshold=float(model_cfg["confidence_threshold"]),
        )

    def predict(self, fv: FeatureVector) -> Prediction:
        """
        Run inference on a single feature vector for live data.

        XGBoost's predict() returns class probabilities when called with
        DMatrix and output_margin=False on a multi-class model, so we take
        argmax for the label and the corresponding probability for confidence.
        """
        x = fv.to_array().reshape(1, -1)
        dmat = xgb.DMatrix(x, feature_names=FeatureVector.feature_names())

        # proba shape: (1, n_classes) for multi-class, (1,) for binary
        proba = self._booster.predict(dmat)
        if proba.ndim == 2:
            label_id = int(np.argmax(proba[0]))
            confidence = float(proba[0, label_id])
        else:
            # binary output: probability of class 1
            confidence = float(proba[0])
            label_id = int(confidence >= 0.5)
            if label_id == 0:
                confidence = 1.0 - confidence

        label = self._labels.get(label_id, f"class_{label_id}")

        return Prediction(
            label=label,
            label_id=label_id,
            confidence=confidence,
            above_threshold=confidence >= self._threshold,
        )
