"""Model files, model inputs, random seeds and device selection.

A model file is a plain dictionary readable with ``torch.load(...,
weights_only=True)``: the class name, the constructor arguments, the weights
and free-form metadata. Loading therefore never executes arbitrary code, which
matters for weights downloaded from the internet.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from galaxy_sed.data import photometry_inputs
from galaxy_sed.models import MODEL_CLASSES

FILE_FORMAT = "galaxy-sed-model-v1"

# File names inside the working directory (``outputs/`` by default). The same
# names are used in ``weights/``, so a working directory can replace it.
PREPARED = "prepared.npz"
PREPROCESSING = "preprocessing.json"
DIRECT_MLP = "direct_mlp.pt"
DIRECT_MLP_PREDICTIONS = "direct_mlp_predictions.npz"
GLO = "glo.pt"
RECONSTRUCTIONS = "reconstructions"
CWGAN = "cwgan.pt"
CWGAN_CRITIC = "cwgan_critic.pt"


def direct_mlp_file(input_mode: str) -> str:
    """``direct_mlp.pt`` reads the photometry, ``direct_mlp_hybrid.pt`` also the reconstructed SED."""
    return DIRECT_MLP if input_mode == "photometry" else f"direct_mlp_{input_mode}.pt"


def direct_mlp_predictions_file(input_mode: str) -> str:
    return (
        DIRECT_MLP_PREDICTIONS
        if input_mode == "photometry"
        else f"direct_mlp_{input_mode}_predictions.npz"
    )


def parameter_heads_file(input_mode: str) -> str:
    return f"parameter_heads_{input_mode}.pt"


def model_inputs(
    data: dict[str, np.ndarray], workdir: Path, input_mode: str, split: str
) -> np.ndarray:
    """Inputs of the parameter networks for one split.

    ``photometry``: the 20 standardised fluxes and their 20 mask bits.
    ``hybrid``: the same, followed by the standardised SED reconstructed by the
    GLO inversion (step 4), read from the working directory.
    """
    photometry = photometry_inputs(data, data[f"idx_{split}"])
    if input_mode == "photometry":
        return photometry
    sed = np.load(Path(workdir) / RECONSTRUCTIONS / f"glo_{split}.npz")["sed"]
    return np.concatenate([photometry, sed.astype(np.float32)], axis=1)


def common_arguments(description: str) -> argparse.ArgumentParser:
    """Argument parser with the options shared by every script."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config", type=Path, default=Path("config.json"), help="configuration file"
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("outputs"),
        help="directory holding the prepared data, models and predictions",
    )
    return parser


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    """Seed every random generator and make GPU convolutions deterministic.

    By default cuDNN may pick convolution algorithms whose results vary from
    run to run, GLO training amplifies those tiny differences until two runs
    with the same seed end in different generators. Deterministic algorithms
    remove the effect at no measurable cost here.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_model(
    path: str | Path,
    model: nn.Module,
    init_kwargs: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save weights with what is needed to rebuild the model."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": FILE_FORMAT,
            "class": type(model).__name__,
            "init_kwargs": init_kwargs,
            "state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "metadata": metadata or {},
        },
        path,
    )


def load_model(
    path: str | Path, device: torch.device | str = "cpu"
) -> tuple[nn.Module, dict[str, Any]]:
    """Rebuild a model saved by ``save_model``, in evaluation mode."""
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format") != FILE_FORMAT:
        raise ValueError(f"{path} is not a galaxy-sed model file")
    model = MODEL_CLASSES[payload["class"]](**payload["init_kwargs"])
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, payload["metadata"]


def write_json(path: str | Path, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
