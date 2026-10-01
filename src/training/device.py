"""
Device selection.

[A: F31] The development machine has no discrete GPU (Intel Arc integrated only;
the RTX 3070 Ti of the v1 runs was a different machine), and a MacBook Pro
(M5 Pro) is a second target. The default is CPU on purpose: a 147->64->64 MLP is
far too small to pay for host-device transfer, and SB3's own guidance is to use
CPU for `MlpPolicy`. `cuda` and `mps` remain reachable by explicit request.
"""

from __future__ import annotations

import torch

VALID = ("auto", "cpu", "cuda", "mps")


def available_accelerator() -> str | None:
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return None


def resolve_device(requested: str = "cpu") -> str:
    """
    'cpu' (default) -> cpu. 'auto' -> cuda, else mps, else cpu.
    'cuda'/'mps' -> that device, or a RuntimeError if it is not there, because a
    silent fallback would quietly invalidate a timing comparison.
    """
    if requested not in VALID:
        raise ValueError(f"device must be one of {VALID}, got {requested!r}")

    if requested == "cpu":
        return "cpu"
    if requested == "auto":
        return available_accelerator() or "cpu"

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "device='cuda' requested but CUDA is not available "
            f"(torch {torch.__version__}). This machine has no discrete GPU."
        )
    if requested == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError(
                f"device='mps' requested but MPS is not available (torch {torch.__version__})."
            )
    return requested


def describe() -> str:
    acc = available_accelerator()
    return (
        f"torch {torch.__version__} | accelerator: {acc or 'none'} | default device: cpu"
    )
