"""Lightweight helpers for batting outcome prediction models."""

from __future__ import annotations

from typing import Iterable, List, Sequence

try:  # pragma: no cover - optional dependency
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
except ImportError:  # pragma: no cover - fallback when PyTorch is unavailable
    torch = None  # type: ignore
    nn = None  # type: ignore

REQUIRED_FEATURES: Sequence[str] = ("K%", "BB%", "Hard%", "GB%")


if nn is not None:

    class Net(nn.Module):
        """Simple feed-forward network used for probability prediction."""

        def __init__(self, input_dim: int = 4, output_dim: int = 4) -> None:
            super().__init__()
            self.fc1 = nn.Linear(input_dim, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, output_dim)

        def forward(self, x):  # type: ignore[override]
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            return self.fc3(x)

else:  # pragma: no cover - PyTorch not installed

    class Net:  # type: ignore[too-few-public-methods]
        """Placeholder used when PyTorch is not available."""

        def __init__(self, *args, **kwargs) -> None:  # noqa: D401 - simple message
            raise RuntimeError("PyTorch is required to instantiate Net")

        def forward(self, x):  # pragma: no cover - never called without torch
            raise RuntimeError("PyTorch is required to use Net")


def _extract_features(sample: dict) -> List[float]:
    """Return the feature vector expected by the prediction models."""

    values: List[float] = []
    for feature in REQUIRED_FEATURES:
        if feature not in sample:
            raise ValueError(f"必須特徴量 '{feature}' が入力に含まれていません。")
        values.append(float(sample[feature]))
    return values


def predict(model, sample: dict) -> Iterable[float]:
    """Predict the outcome distribution using a scikit-learn compatible model."""

    if not hasattr(model, "predict"):
        raise AttributeError("Model does not provide a predict() method")
    feature_vector = _extract_features(sample)
    prediction = model.predict([feature_vector])
    return prediction[0]


def predict_linear(model, sample: dict) -> Iterable[float]:
    """Alias kept for backwards compatibility with legacy callers."""

    return predict(model, sample)


def predict_nn(model, sample: dict) -> Iterable[float]:
    """Run the neural network model if PyTorch is available."""

    if torch is None:
        raise RuntimeError("PyTorch is required for neural network predictions")

    feature_vector = _extract_features(sample)
    tensor_input = torch.tensor([feature_vector], dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        prediction = model(tensor_input)

    result = prediction.squeeze(0)
    if hasattr(result, "tolist"):
        return result.tolist()
    return [float(value) for value in result]


def predict_auto(model, sample: dict, model_type: str = "linear") -> Iterable[float]:
    """Dispatch prediction to the appropriate helper based on model type."""

    if model_type == "linear":
        return predict_linear(model, sample)
    if model_type == "nn":
        return predict_nn(model, sample)
    raise ValueError(f"Unsupported model type: {model_type}")


__all__ = ["Net", "predict", "predict_linear", "predict_nn", "predict_auto"]
