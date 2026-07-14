"""
Optional TPU support for MedSAM3 via AutoXLA.

This module is the single integration point between MedSAM3 and
torch_xla/AutoXLA (https://github.com/Locutusque/autoxla). Everything here is
import-guarded: on machines without torch_xla the rest of the codebase works
exactly as before, and TPU mode fails with a clear error message only when
explicitly requested.

Usage from the training script:

    import tpu_utils

    if tpu_utils.is_tpu_available():
        device = tpu_utils.get_xla_device()
        model = build_sam3_image_model(device="cpu", ...)   # build on host
        model = apply_lora_to_model(model, lora_config)
        model = tpu_utils.prepare_model_for_tpu(model, config.get("tpu", {}))

    # training loop
    loss.backward()
    optimizer.step()
    tpu_utils.mark_step()   # no-op off-TPU

Install requirements on a TPU VM:

    pip install torch~=2.8.0
    pip install 'torch_xla[tpu]~=2.8.0' \
        --find-links=https://storage.googleapis.com/libtpu-releases/index.html \
        --find-links=https://storage.googleapis.com/libtpu-wheels/index.html
    pip install 'torch_xla[pallas]' \
        --find-links=https://storage.googleapis.com/jax-releases/jax_nightly_releases.html \
        --find-links=https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html
    git clone https://github.com/Locutusque/autoxla && pip install -e autoxla
"""

import contextlib
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Guarded imports
# ---------------------------------------------------------------------------

TORCH_XLA_AVAILABLE = False
AUTOXLA_AVAILABLE = False
_IMPORT_ERROR = None

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr

    TORCH_XLA_AVAILABLE = True
except ImportError as e:  # pragma: no cover - exercised only off-TPU
    _IMPORT_ERROR = e

if TORCH_XLA_AVAILABLE:
    try:
        from AutoXLA import AutoXLAModelForImageSegmentation
        from AutoXLA.quantization import QuantizationConfig

        AUTOXLA_AVAILABLE = True
    except ImportError as e:  # pragma: no cover
        _IMPORT_ERROR = e


def is_tpu_available() -> bool:
    """True when torch_xla and AutoXLA are importable (a TPU runtime may still
    be required to actually execute; torch_xla raises at device creation if
    none is configured)."""
    return TORCH_XLA_AVAILABLE and AUTOXLA_AVAILABLE


def require_tpu():
    """Raise a descriptive error when TPU support was requested but the
    dependencies are missing."""
    if not TORCH_XLA_AVAILABLE:
        raise RuntimeError(
            "TPU mode requested but torch_xla is not installed. "
            "See the install snippet in tpu_utils.py. "
            f"(import error: {_IMPORT_ERROR})"
        )
    if not AUTOXLA_AVAILABLE:
        raise RuntimeError(
            "TPU mode requested but AutoXLA is not installed. "
            "Install it with: git clone https://github.com/Locutusque/autoxla "
            "&& pip install -e autoxla "
            f"(import error: {_IMPORT_ERROR})"
        )


def get_xla_device() -> torch.device:
    """Return the XLA device for this process."""
    require_tpu()
    return xm.xla_device()


# ---------------------------------------------------------------------------
# Model preparation
# ---------------------------------------------------------------------------

def _build_quantization_config(quant_cfg: dict) -> "QuantizationConfig":
    """Translate the `tpu.quantization` YAML section into an AutoXLA config.

    The frozen SAM3 base weights are the quantization target (QLoRA-style);
    LoRA adapters stay in full precision because the quantizer only replaces
    nn.Linear modules and the adapters are raw nn.Parameter matmuls.
    """
    exclude = list(quant_cfg.get("exclude_layers", []))
    return QuantizationConfig(
        n_bits=int(quant_cfg.get("n_bits", 8)),
        symmetric=bool(quant_cfg.get("symmetric", True)),
        use_pallas=bool(quant_cfg.get("use_pallas", True)),
        quantize_activation=bool(quant_cfg.get("quantize_activation", False)),
        block_size=int(quant_cfg.get("block_size", -1)),
        exclude_layers=exclude,
    )


def prepare_model_for_tpu(model: nn.Module, tpu_config: dict | None = None) -> nn.Module:
    """Move a (LoRA-augmented) SAM3 model to TPU via AutoXLA.

    Applies AutoXLA's quantize -> shard -> (optional) FSDPv2 pipeline. Must be
    called AFTER apply_lora_to_model (so LoRA modules get sharded/replicated
    consistently with the base weights) and BEFORE the optimizer is created
    (parameters are re-materialized on the XLA device).

    tpu_config keys (all optional):
        sharding_strategy: "fsdp" | "dp" | "mp" | "2d" | "3d"   (default "fsdp")
        use_fsdp_wrap: wrap with torch_xla FSDPv2 (default False; keep off for
            LoRA training so save_lora_weights sees unwrapped module names)
        quantize_base_model: int8/int4-quantize the frozen base linears
            (QLoRA-style) via AutoXLA (default False)
        quantization: dict forwarded to AutoXLA's QuantizationConfig
    """
    require_tpu()
    tpu_config = tpu_config or {}

    do_quant = bool(tpu_config.get("quantize_base_model", False))
    quantization_config = None
    if do_quant:
        quantization_config = _build_quantization_config(tpu_config.get("quantization", {}))
        logger.info("Quantizing frozen base model weights (QLoRA-style) via AutoXLA")

    return AutoXLAModelForImageSegmentation.from_model(
        model,
        sharding_strategy=tpu_config.get("sharding_strategy", "fsdp"),
        use_fsdp_wrap=bool(tpu_config.get("use_fsdp_wrap", False)),
        do_quant=do_quant,
        quantization_config=quantization_config,
        verbose=bool(tpu_config.get("verbose", True)),
    )


# ---------------------------------------------------------------------------
# Training-loop helpers (no-ops off TPU so call sites stay unconditional)
# ---------------------------------------------------------------------------

def mark_step():
    """Cut the lazy XLA graph and dispatch it for execution.

    Call once per optimizer step (and per eval batch). A no-op off TPU.
    """
    if TORCH_XLA_AVAILABLE:
        xm.mark_step()


def autocast_context(device: torch.device, enabled: bool):
    """bfloat16 autocast for XLA, mirroring `training.mixed_precision: bf16`.

    Returns a nullcontext when disabled or off-TPU.
    """
    if not enabled or not TORCH_XLA_AVAILABLE or device.type != "xla":
        return contextlib.nullcontext()
    try:
        from torch_xla.amp import autocast as xla_autocast

        return xla_autocast(device, dtype=torch.bfloat16)
    except (ImportError, TypeError):
        return torch.autocast(device_type="xla", dtype=torch.bfloat16)


def world_size() -> int:
    """Number of addressable TPU devices (1 off-TPU). With SPMD sharding a
    single process drives all devices, so this is informational only."""
    if not TORCH_XLA_AVAILABLE:
        return 1
    try:
        return xr.global_runtime_device_count()
    except Exception:
        return 1
