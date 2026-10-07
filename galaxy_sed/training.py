"""Training loops for the four models.

Each function takes the prepared dataset (``galaxy_sed.data.load_prepared``),
its section of the configuration and a device, and returns the trained model
with a per-epoch history. Selection follows the reference study: the direct
MLP, the parameter heads and the cWGAN-GP keep the epoch with the best
validation score, the GLO generator, which has no validation objective, keeps
the epoch with the lowest training loss.

The random generators are seeded at the start of every function, so a given
configuration always starts from the same state.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from galaxy_sed.data import parameter_mask, photometry_inputs
from galaxy_sed.io import seed_everything
from galaxy_sed.models import (
    CWGANCritic,
    CWGANGenerator,
    DirectMLP,
    GLOGenerator,
    LatentCodes,
    ParameterHeads,
)

Logger = Callable[[str], None]

# Minimum improvement counted as progress by the early-stopping rules.
VALIDATION_MIN_DELTA = 1e-5
GLO_MIN_DELTA = 1e-4
# Range of the log variance predicted by the uncertainty heads.
LOG_VAR_RANGE = (-7.0, 7.0)


def masked_mse(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Mean squared error over the entries where ``mask`` is 1."""
    return torch.sum(torch.square(prediction - target) * mask) / torch.clamp_min(
        torch.sum(mask), 1.0
    )


def _state_copy(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


# --------------------------------------------------------------------------- #
# Direct MLP
# --------------------------------------------------------------------------- #
def train_direct_mlp(
    data: dict[str, np.ndarray],
    cfg: dict[str, Any],
    device: torch.device,
    log: Logger = print,
    inputs_train: np.ndarray | None = None,
    inputs_val: np.ndarray | None = None,
) -> tuple[DirectMLP, list[dict[str, float]]]:
    """One network for all parameters, masked MSE, early stopping on validation.

    The inputs are the 40 photometric inputs unless ``inputs_train`` and
    ``inputs_val`` are given, for instance the photometry followed by the
    reconstructed SED (hybrid input, see ``galaxy_sed.io.model_inputs``).
    """
    seed_everything(cfg["seed"])
    idx_train, idx_val = data["idx_train"], data["idx_val"]
    if inputs_train is None:
        inputs_train = photometry_inputs(data, idx_train)
    if inputs_val is None:
        inputs_val = photometry_inputs(data, idx_val)
    x_train = torch.tensor(inputs_train, dtype=torch.float32)
    y_train = torch.tensor(data["Y_params"][idx_train], dtype=torch.float32)
    m_train = torch.tensor(parameter_mask(data, idx_train))
    x_val = torch.tensor(inputs_val, dtype=torch.float32, device=device)
    y_val = torch.tensor(data["Y_params"][idx_val], dtype=torch.float32, device=device)
    m_val = torch.tensor(parameter_mask(data, idx_val), device=device)

    loader = DataLoader(
        TensorDataset(x_train, y_train, m_train),
        batch_size=cfg["batch_size"],
        shuffle=True,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(cfg["seed"]),
    )
    model = DirectMLP(
        n_in=x_train.shape[1], n_out=y_train.shape[1], dropout=cfg["dropout"]
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    best_val, best_state, stale, history = float("inf"), None, 0, []
    for epoch in range(1, cfg["epochs"] + 1):
        started = time.time()
        model.train()
        losses = []
        for inputs, targets, mask in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            loss = masked_mse(model(inputs), targets, mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            val_loss = float(masked_mse(model(x_val), y_val, m_val).item())
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "val_loss": val_loss,
                "time_sec": time.time() - started,
            }
        )
        log(
            f"[{epoch:3d}/{cfg['epochs']}] train={history[-1]['train_loss']:.5f} val={val_loss:.5f}"
        )
        if val_loss < best_val - VALIDATION_MIN_DELTA:
            best_val, best_state, stale = val_loss, _state_copy(model), 0
        else:
            stale += 1
        if stale >= cfg["patience"]:
            log(f"Early stopping at epoch {epoch}")
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, history


# --------------------------------------------------------------------------- #
# GLO generator
# --------------------------------------------------------------------------- #
class UncertaintyWeightedLoss(nn.Module):
    """SED and parameter reconstruction losses balanced by learned task scales.

    Each task gets a learned log scale ``s`` and contributes
    ``0.5 * exp(-2 s) * loss + s``, so neither objective has to be weighted by
    hand (homoscedastic uncertainty weighting, Kendall et al. 2018).
    """

    def __init__(self) -> None:
        super().__init__()
        self.log_sigma_sed = nn.Parameter(torch.zeros(()))
        self.log_sigma_params = nn.Parameter(torch.zeros(()))

    def forward(self, sed_pred, sed_true, params_pred, params_true, params_mask):
        sed_loss = F.mse_loss(sed_pred, sed_true)
        params_loss = masked_mse(params_pred, params_true, params_mask)
        log_sed = self.log_sigma_sed.clamp(-3.0, 3.0)
        log_params = self.log_sigma_params.clamp(-3.0, 3.0)
        total = (
            0.5 * torch.exp(-2.0 * log_sed) * sed_loss
            + log_sed
            + 0.5 * torch.exp(-2.0 * log_params) * params_loss
            + log_params
        )
        return total, sed_loss, params_loss


def train_glo(
    data: dict[str, np.ndarray],
    cfg: dict[str, Any],
    device: torch.device,
    log: Logger = print,
) -> tuple[GLOGenerator, list[dict[str, float]]]:
    """Generative latent optimisation.

    The generator and one latent code per training galaxy are optimised
    jointly to reproduce each galaxy's SED and parameters. The codes stay on
    the unit sphere so the generator learns a bounded, well-behaved latent
    space that can later be searched by inversion.
    """
    seed_everything(cfg["seed"])
    idx = data["idx_train"]
    sed = torch.tensor(data["Y_sed"][idx], dtype=torch.float32)
    params = torch.tensor(data["Y_params"][idx], dtype=torch.float32)
    mask = torch.tensor(parameter_mask(data, idx))
    loader = DataLoader(
        TensorDataset(torch.arange(len(idx)), sed, params, mask),
        batch_size=cfg["batch_size"],
        shuffle=True,
        pin_memory=device.type == "cuda",
    )
    model = GLOGenerator(
        latent_dim=cfg["latent_dim"], sed_len=sed.shape[1], n_params=params.shape[1]
    ).to(device)
    codes = LatentCodes(len(idx), cfg["latent_dim"], cfg["seed"]).to(device)
    loss_fn = UncertaintyWeightedLoss().to(device)
    model_optimizer = torch.optim.Adam(
        list(model.parameters()) + list(loss_fn.parameters()),
        lr=cfg["lr_generator"],
        betas=(0.5, 0.999),
    )
    code_optimizer = torch.optim.Adam(
        codes.parameters(), lr=cfg["lr_latent"], betas=(0.5, 0.999)
    )
    clipped = [p for group in model_optimizer.param_groups for p in group["params"]]

    best_total, best_state, stale, history = float("inf"), None, 0, []
    for epoch in range(1, cfg["epochs"] + 1):
        started = time.time()
        model.train()
        sums = {"total": 0.0, "sed": 0.0, "params": 0.0}
        batches = 0
        for indices, sed_true, params_true, params_mask in loader:
            indices = indices.to(device)
            sed_true = sed_true.to(device, non_blocking=True)
            params_true = params_true.to(device, non_blocking=True)
            params_mask = params_mask.to(device, non_blocking=True)
            sed_pred, params_pred = model(codes(indices))
            total, sed_loss, params_loss = loss_fn(
                sed_pred, sed_true, params_pred, params_true, params_mask
            )
            model_optimizer.zero_grad(set_to_none=True)
            code_optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(clipped, max_norm=5.0)
            torch.nn.utils.clip_grad_norm_(codes.parameters(), max_norm=5.0)
            model_optimizer.step()
            code_optimizer.step()
            with torch.no_grad():  # project the updated codes back on the sphere
                selected = codes.codes.data[indices]
                codes.codes.data[indices] = selected / selected.norm(
                    dim=1, keepdim=True
                ).clamp(min=1e-12)
            sums["total"] += float(total.item())
            sums["sed"] += float(sed_loss.item())
            sums["params"] += float(params_loss.item())
            batches += 1
        record = {
            "epoch": epoch,
            **{key: value / batches for key, value in sums.items()},
            "time_sec": time.time() - started,
        }
        history.append(record)
        log(
            f"[{epoch:3d}/{cfg['epochs']}] total={record['total']:+.5f} sed={record['sed']:.5f} "
            f"params={record['params']:.5f}"
        )
        if record["total"] < best_total - GLO_MIN_DELTA:
            best_total, best_state, stale = record["total"], _state_copy(model), 0
        else:
            stale += 1
        if stale >= cfg["patience"]:
            log(f"Early stopping at epoch {epoch}")
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, history


# --------------------------------------------------------------------------- #
# Per-parameter heads
# --------------------------------------------------------------------------- #
def classification_targets(
    params_std: np.ndarray, specs: list[dict[str, Any]]
) -> np.ndarray:
    """Index of the nearest grid level for every classification parameter."""
    targets = np.zeros(params_std.shape, dtype=np.int64)
    for spec in specs:
        if spec["type"] == "cls":
            column = params_std[:, spec["index"]].astype(np.float64)[:, None]
            levels = np.asarray(spec["levels_std"], dtype=np.float64)[None, :]
            targets[:, spec["index"]] = np.argmin(np.abs(column - levels), axis=1)
    return targets


def heads_loss(
    outputs: list[torch.Tensor],
    params_std: torch.Tensor,
    classes: torch.Tensor,
    mask: torch.Tensor,
    specs: list[dict[str, Any]],
    uncertainty: bool,
) -> torch.Tensor:
    """Sum over parameters of the masked loss of each head.

    Cross entropy for classification heads, Gaussian negative log likelihood
    ``0.5 * (log var + (y - mu)^2 / var)`` for regression heads with
    uncertainty, plain squared error otherwise.
    """
    total = params_std.new_zeros(())
    for k, spec in enumerate(specs):
        j = spec["index"]
        weight = mask[:, j]
        denom = torch.clamp_min(weight.sum(), 1.0)
        if spec["type"] == "cls":
            loss = F.cross_entropy(outputs[k], classes[:, j], reduction="none")
        elif uncertainty:
            mean, log_var = outputs[k][:, 0], outputs[k][:, 1].clamp(*LOG_VAR_RANGE)
            loss = 0.5 * (
                log_var + torch.square(params_std[:, j] - mean) / torch.exp(log_var)
            )
        else:
            loss = torch.square(outputs[k].squeeze(1) - params_std[:, j])
        total = total + torch.sum(loss * weight) / denom
    return total


def train_parameter_heads(
    data: dict[str, np.ndarray],
    inputs_train: np.ndarray,
    inputs_val: np.ndarray,
    specs: list[dict[str, Any]],
    cfg: dict[str, Any],
    device: torch.device,
    log: Logger = print,
) -> tuple[ParameterHeads, list[dict[str, float]]]:
    """Train one small network per parameter, all in the same loop.

    The heads share no weights, so this is equivalent to training them
    separately, batching them only saves time.
    """
    seed_everything(cfg["seed"])
    idx_train, idx_val = data["idx_train"], data["idx_val"]
    x_train = torch.tensor(inputs_train, dtype=torch.float32)
    y_train = torch.tensor(data["Y_params"][idx_train], dtype=torch.float32)
    m_train = torch.tensor(parameter_mask(data, idx_train))
    c_train = torch.tensor(classification_targets(data["Y_params"][idx_train], specs))
    x_val = torch.tensor(inputs_val, dtype=torch.float32, device=device)
    y_val = torch.tensor(data["Y_params"][idx_val], dtype=torch.float32, device=device)
    m_val = torch.tensor(parameter_mask(data, idx_val), device=device)
    c_val = torch.tensor(
        classification_targets(data["Y_params"][idx_val], specs), device=device
    )

    loader = DataLoader(
        TensorDataset(x_train, y_train, m_train, c_train),
        batch_size=cfg["batch_size"],
        shuffle=True,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(cfg["seed"]),
    )
    model = ParameterHeads(
        x_train.shape[1], specs, uncertainty=cfg["uncertainty"], dropout=cfg["dropout"]
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    def loss_of(x, y, m, c):
        return heads_loss(model(x), y, c, m, specs, cfg["uncertainty"])

    best_val, best_state, stale, history = float("inf"), None, 0, []
    for epoch in range(1, cfg["epochs"] + 1):
        started = time.time()
        model.train()
        losses = []
        for xb, yb, mb, cb in loader:
            loss = loss_of(
                xb.to(device, non_blocking=True),
                yb.to(device, non_blocking=True),
                mb.to(device, non_blocking=True),
                cb.to(device, non_blocking=True),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_of(x_val, y_val, m_val, c_val).item())
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "val_loss": val_loss,
                "time_sec": time.time() - started,
            }
        )
        log(
            f"[{epoch:3d}/{cfg['epochs']}] train={history[-1]['train_loss']:.5f} val={val_loss:.5f}"
        )
        if val_loss < best_val - VALIDATION_MIN_DELTA:
            best_val, best_state, stale = val_loss, _state_copy(model), 0
        else:
            stale += 1
        if stale >= cfg["patience"]:
            log(f"Early stopping at epoch {epoch}")
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, history


# --------------------------------------------------------------------------- #
# Conditional WGAN-GP
# --------------------------------------------------------------------------- #
def gradient_penalty(
    critic, real_sed, real_params, fake_sed, fake_params, cond
) -> torch.Tensor:
    """WGAN-GP penalty on random interpolates between real and generated samples."""
    epsilon = torch.rand(real_sed.shape[0], 1, device=real_sed.device)
    sed_mix = (epsilon * real_sed + (1.0 - epsilon) * fake_sed).requires_grad_(True)
    params_mix = (epsilon * real_params + (1.0 - epsilon) * fake_params).requires_grad_(
        True
    )
    score = critic(sed_mix, params_mix, cond)
    gradients = torch.autograd.grad(
        score,
        [sed_mix, params_mix],
        grad_outputs=torch.ones_like(score),
        create_graph=True,
        retain_graph=True,
    )
    flat = torch.cat([gradients[0].flatten(1), gradients[1].flatten(1)], dim=1)
    return torch.mean(torch.square(torch.linalg.vector_norm(flat, dim=1) - 1.0))


def _masked_smooth_l1(prediction, target, mask) -> torch.Tensor:
    losses = F.smooth_l1_loss(prediction, target, reduction="none")
    return torch.sum(losses * mask) / torch.clamp_min(torch.sum(mask), 1.0)


@torch.no_grad()
def _validate_cwgan(
    generator, loader, device, polar_columns, latent_dim, seed, draws
) -> dict[str, float]:
    """Validation score: SED L1 plus masked parameter Smooth-L1 of the draw average.

    The same noise is used at every epoch, so scores are comparable.
    """
    generator.eval()
    noise_generator = torch.Generator(device=device).manual_seed(seed)
    totals = {"sed_l1": 0.0, "params_smooth_l1": 0.0}
    count = 0
    for real_sed, real_params, cond, flags in loader:
        real_sed, real_params = real_sed.to(device), real_params.to(device)
        cond, flags = cond.to(device), flags.to(device)
        seds, params = [], []
        for _ in range(draws):
            noise = torch.randn(
                cond.shape[0], latent_dim, generator=noise_generator, device=device
            )
            sed, param = generator(noise, cond)
            seds.append(sed)
            params.append(param)
        mask = torch.ones_like(real_params)
        for column in polar_columns:
            mask[:, column] = flags
        batch = real_sed.shape[0]
        totals["sed_l1"] += (
            F.l1_loss(torch.stack(seds).mean(0), real_sed).item() * batch
        )
        totals["params_smooth_l1"] += (
            _masked_smooth_l1(torch.stack(params).mean(0), real_params, mask).item()
            * batch
        )
        count += batch
    scores = {key: value / max(count, 1) for key, value in totals.items()}
    scores["selection_score"] = scores["sed_l1"] + scores["params_smooth_l1"]
    return scores


def train_cwgan(
    data: dict[str, np.ndarray],
    cfg: dict[str, Any],
    device: torch.device,
    log: Logger = print,
) -> tuple[CWGANGenerator, CWGANCritic, list[dict[str, float]]]:
    """Conditional Wasserstein GAN with gradient penalty.

    The generator receives noise and the photometry and must produce an SED
    and parameters that the critic cannot tell apart from real ones for that
    photometry. The critic is updated ``n_critic`` times per generator update.
    """
    seed_everything(cfg["seed"])
    idx_train, idx_val = data["idx_train"], data["idx_val"]
    names = [str(name) for name in data["param_names"]]
    polar_columns = [
        names.index(str(name)) for name in data["polar_dependent"] if str(name) in names
    ]
    latent_dim = cfg["latent_dim"]

    train_sed = torch.tensor(data["Y_sed"][idx_train], dtype=torch.float32)
    train_params = torch.tensor(data["Y_params"][idx_train], dtype=torch.float32)
    train_cond = torch.tensor(photometry_inputs(data, idx_train))
    train_loader = DataLoader(
        TensorDataset(train_sed, train_params, train_cond),
        batch_size=cfg["batch_size"],
        shuffle=True,
        drop_last=True,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(cfg["seed"]),
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(data["Y_sed"][idx_val], dtype=torch.float32),
            torch.tensor(data["Y_params"][idx_val], dtype=torch.float32),
            torch.tensor(photometry_inputs(data, idx_val)),
            torch.tensor(data["Y_flag_pol"][idx_val], dtype=torch.float32),
        ),
        batch_size=max(cfg["batch_size"], 256),
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    sizes = {"sed_len": train_sed.shape[1], "n_params": train_params.shape[1]}
    generator = CWGANGenerator(
        latent_dim=latent_dim, cond_dim=train_cond.shape[1], **sizes
    ).to(device)
    critic = CWGANCritic(cond_dim=train_cond.shape[1], **sizes).to(device)
    betas = tuple(cfg["betas"])
    optimizer_g = torch.optim.Adam(
        generator.parameters(), lr=cfg["lr_generator"], betas=betas
    )
    optimizer_d = torch.optim.Adam(
        critic.parameters(), lr=cfg["lr_critic"], betas=betas
    )

    best_score, best_generator, best_critic, history = float("inf"), None, None, []
    for epoch in range(1, cfg["epochs"] + 1):
        started = time.time()
        generator.train()
        critic.train()
        totals = {"critic": 0.0, "generator": 0.0, "wasserstein": 0.0}
        groups = 0
        batches = iter(train_loader)
        exhausted = False
        while not exhausted:
            # Critic updates on real batches.
            for _ in range(cfg["n_critic"]):
                try:
                    real_sed, real_params, cond = next(batches)
                except StopIteration:
                    exhausted = True
                    break
                real_sed = real_sed.to(device, non_blocking=True)
                real_params = real_params.to(device, non_blocking=True)
                cond = cond.to(device, non_blocking=True)
                noise = torch.randn(real_sed.shape[0], latent_dim, device=device)
                with torch.no_grad():
                    fake_sed, fake_params = generator(noise, cond)
                real_score = critic(real_sed, real_params, cond).mean()
                fake_score = critic(fake_sed, fake_params, cond).mean()
                penalty = gradient_penalty(
                    critic, real_sed, real_params, fake_sed, fake_params, cond
                )
                critic_loss = (
                    fake_score - real_score + cfg["gradient_penalty"] * penalty
                )
                optimizer_d.zero_grad(set_to_none=True)
                critic_loss.backward()
                optimizer_d.step()
            if exhausted:
                break
            # One generator update on randomly drawn photometry.
            picks = torch.randint(0, train_cond.shape[0], (cfg["batch_size"],))
            gen_cond = train_cond[picks].to(device, non_blocking=True)
            noise = torch.randn(cfg["batch_size"], latent_dim, device=device)
            gen_sed, gen_params = generator(noise, gen_cond)
            generator_loss = -critic(gen_sed, gen_params, gen_cond).mean()
            optimizer_g.zero_grad(set_to_none=True)
            generator_loss.backward()
            optimizer_g.step()
            totals["critic"] += critic_loss.item()
            totals["generator"] += generator_loss.item()
            totals["wasserstein"] += (real_score - fake_score).item()
            groups += 1

        scores = _validate_cwgan(
            generator,
            val_loader,
            device,
            polar_columns,
            latent_dim,
            cfg["seed"],
            cfg["validation_draws"],
        )
        record = {
            "epoch": epoch,
            **{f"train_{k}": v / max(groups, 1) for k, v in totals.items()},
            **{f"val_{k}": v for k, v in scores.items()},
            "time_sec": time.time() - started,
        }
        history.append(record)
        log(
            f"[{epoch:3d}/{cfg['epochs']}] critic={record['train_critic']:+.3f} "
            f"generator={record['train_generator']:+.3f} val={scores['selection_score']:.4f}"
        )
        if scores["selection_score"] < best_score:
            best_score = scores["selection_score"]
            best_generator, best_critic = _state_copy(generator), _state_copy(critic)
    generator.load_state_dict(best_generator)
    critic.load_state_dict(best_critic)
    generator.eval()
    critic.eval()
    return generator, critic, history
