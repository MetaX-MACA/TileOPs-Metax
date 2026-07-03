import functools

import torch

str2dtype = {
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
    'float32': torch.float32,
    "int32": torch.int32
}

dtype2str = {v: k for k, v in str2dtype.items()}


@torch.compile
def reduce_on_dim0(x: torch.Tensor) -> torch.Tensor:
    """Reduce a tensor on dimension 0.

    Arguments:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Reduced tensor.
    """
    return x[0] if x.size(0) == 1 else x.sum(dim=0)


@torch.compile
def zero_pad(x: torch.Tensor, pad_size: int, dim: int) -> torch.Tensor:
    """Pad a tensor with 0 to a be divisible by `pad_size` along a specified dimension.

    Arguments:
        x (torch.Tensor): Input tensor.
        pad_size (int): The size to pad to be divisible by.
        dim (int): The dimension to pad.

    Returns:
        torch.Tensor: Padded tensor.
    """
    if x.size(dim) % pad_size == 0:
        return x
    pad_len = (pad_size - x.size(dim) % pad_size)
    assert 0 < pad_len < pad_size

    zero_shape = list(x.shape)
    zero_shape[dim] = pad_len
    zero_shape = tuple(zero_shape)
    zeros = torch.zeros(zero_shape, dtype=x.dtype, device=x.device)
    return torch.cat((x, zeros), dim=dim)


def ensure_contiguous(func: callable) -> callable:
    """Decorator to ensure that all tensor arguments are contiguous before calling the function.

    Arguments:
        func (callable): The function to decorate.

    Returns:
        callable: The decorated function.
    """

    def wrapper(*args, **kwargs):
        args = [arg.contiguous() if isinstance(arg, torch.Tensor) else arg for arg in args]
        kwargs = {
            k: v.contiguous() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()
        }
        return func(*args, **kwargs)

    return wrapper


def is_hopper():
    return torch.cuda.get_device_capability() == (9, 0)


@functools.lru_cache(maxsize=1)
def is_h200():
    if not torch.cuda.is_available():
        return False
    return "H200" in torch.cuda.get_device_name().upper()


def get_sm_version():
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


# ---------------------------------------------------------------------------
# Compile-target detection and shared memory budgets (MACA/CUDA)
# ---------------------------------------------------------------------------

_DEFAULT_SMEM_BYTES = 48 * 1024
_MACA_SMEM_BYTES = 65536


@functools.lru_cache(maxsize=1)
def get_compile_target():
    from tilelang.utils.target import determine_target

    return determine_target("auto", return_object=True)


def is_maca() -> bool:
    """Return True when the compile target is MetaX MACA."""
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
