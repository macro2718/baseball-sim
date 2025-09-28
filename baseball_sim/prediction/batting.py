"""Utilities for loading batting prediction models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:  # pragma: no cover - optional dependency
    import joblib  # type: ignore
except ImportError:  # pragma: no cover - graceful degradation
    joblib = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import torch  # type: ignore
except ImportError:  # pragma: no cover - graceful degradation
    torch = None  # type: ignore

from prediction_models.prediction import Net


@dataclass
class BattingModel:
    """Container for the batting model and its type."""

    estimator: Optional[object]
    model_type: str


class BattingModelLoader:
    """Loads the configured batting prediction model with graceful fallbacks."""

    def __init__(self, config, path_manager, logger) -> None:
        self._config = config
        self._path_manager = path_manager
        self._logger = logger

    def load(self) -> BattingModel:
        """Load the batting model based on the project configuration."""
        model_type = self._config.get("simulation.prediction_model_type", "linear")
        try:
            if model_type == "linear":
                return self._load_linear_model()
            if model_type == "nn":
                return self._load_nn_model()
            self._logger.error(f"Unknown model type requested: {model_type}")
        except Exception as exc:  # pragma: no cover - defensive logging wrapper
            self._logger.error(f"Failed to load batting model ({model_type}): {exc}")
        return BattingModel(estimator=None, model_type="linear")

    def _load_linear_model(self) -> BattingModel:
        model_path = self._path_manager.get_batting_model_path()
        if not self._path_manager.file_exists(model_path) or joblib is None:
            if joblib is None:
                self._logger.warning(
                    "joblib is not installed; using default linear prediction model"
                )
            else:
                self._logger.warning(
                    f"Linear batting model not found at {model_path}, using default prediction"
                )
            return BattingModel(estimator=None, model_type="linear")

        model_info = joblib.load(model_path)
        estimator = model_info.get("model") if isinstance(model_info, dict) else model_info
        self._logger.info("Linear batting model loaded successfully")
        return BattingModel(estimator=estimator, model_type="linear")

    def _load_nn_model(self) -> BattingModel:
        if torch is None:
            self._logger.warning(
                "PyTorch is not installed; falling back to linear prediction model"
            )
            return self._load_linear_model()

        model_path = self._path_manager.get_nn_model_path()
        if not self._path_manager.file_exists(model_path):
            self._logger.warning(
                f"NN model not found at {model_path}, falling back to linear model"
            )
            return self._load_linear_model()

        # モデルの最終層形状から出力次元(4 or 5)を自動判別
        state_dict = torch.load(model_path, map_location="cpu")
        out_dim = 5
        try:
            weight = state_dict.get("fc3.weight")
            if weight is not None and hasattr(weight, "shape") and len(weight.shape) >= 1:
                out_dim = int(weight.shape[0])
        except Exception:
            out_dim = 5

        model = Net(input_dim=4, output_dim=out_dim)
        model.load_state_dict(state_dict)
        model.eval()
        self._logger.info("NN batting model loaded successfully")
        return BattingModel(estimator=model, model_type="nn")
