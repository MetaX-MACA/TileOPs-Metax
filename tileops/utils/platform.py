"""Compile-target detection and memory budgets for MACA/CUDA kernel decisions."""

from __future__ import annotations

import functools

_DEFAULT_SMEM_BYTES = 48 * 1024
_MACA_SMEM_BYTES = 65536


@functools.lru_cache(maxsize=1)
def get_compile_target():
    from tilelang.utils.target import determine_target

    return determine_target("auto", return_object=True)


def is_maca() -> bool:
    """Return True when the compile target is MetaX MACA."""
    import torch

    version = getattr(torch, "version", None)
    if version is not None:
        ver = getattr(version, "__version__", "") or ""
        if "metax" in ver or hasattr(version, "maca"):
            return True

    try:
        from tilelang.utils.target import target_is_maca

        return target_is_maca(get_compile_target())
    except Exception:
        return False


def shared_memory_budget_bytes(device_index: int | None = None) -> int:
    """Return per-block shared memory budget in bytes for tiling/guards."""
    if is_maca():
        target = get_compile_target()
        return int(target.attrs.get("max_shared_memory_per_block", _MACA_SMEM_BYTES))

    import torch

    try:
        if not torch.cuda.is_available():
            return _DEFAULT_SMEM_BYTES
        if device_index is None:
            device_index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_index)
        smem_optin = getattr(props, "shared_memory_per_block_optin", 0)
        if smem_optin > 0:
            return smem_optin
        return getattr(props, "shared_memory_per_block", _DEFAULT_SMEM_BYTES)
    except Exception:
        return _DEFAULT_SMEM_BYTES
