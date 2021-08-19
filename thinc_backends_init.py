import contextlib
from typing import Type, Dict, Any

from contextvars import ContextVar
import threading

from .ops import Ops
from .cupy_ops import CupyOps, has_cupy
from .numpy_ops import NumpyOps
from ._cupy_allocators import cupy_tensorflow_allocator, cupy_pytorch_allocator
from ._param_server import ParamServer
from ..util import assert_tensorflow_installed, assert_pytorch_installed
from ..util import is_cupy_array
from ..types import OpsNames


context_ops: ContextVar[NumpyOps] = ContextVar("context_ops", default=NumpyOps())
context_pools: ContextVar[dict] = ContextVar("context_pools", default={})

_GLOBAL_STATE = {"ops": NumpyOps()}


def set_gpu_allocator(allocator: str) -> None:
    if allocator == "pytorch":
        use_pytorch_for_gpu_memory()
    elif allocator == "tensorflow":
        use_tensorflow_for_gpu_memory()
    else:
        raise ValueError(
            f"Invalid 'gpu_allocator' argument: '{allocator}'. Available allocators are: 'pytorch', 'tensorflow'"
        )


def use_pytorch_for_gpu_memory() -> None:
    import cupy.cuda

    assert_pytorch_installed()
    pools = context_pools.get()
    if "pytorch" not in pools:
        pools["pytorch"] = cupy.cuda.MemoryPool(allocator=cupy_pytorch_allocator)
    cupy.cuda.set_allocator(pools["pytorch"].malloc)


def use_tensorflow_for_gpu_memory() -> None:
    import cupy.cuda

    assert_tensorflow_installed()
    pools = context_pools.get()
    if "tensorflow" not in pools:
        pools["tensorflow"] = cupy.cuda.MemoryPool(allocator=cupy_tensorflow_allocator)
    cupy.cuda.set_allocator(pools["tensorflow"].malloc)


def get_ops(name: OpsNames, **kwargs) -> Ops:
    ops = {"numpy": NumpyOps, "cupy": CupyOps}
    if name not in ops:
        raise ValueError(f"Invalid backend: {name}")
    cls = ops[name]
    return cls(**kwargs)


def get_array_ops(arr):
    if is_cupy_array(arr):
        return CupyOps()
    else:
        return NumpyOps()


@contextlib.contextmanager
def use_ops(name: OpsNames, **kwargs):
    current_ops = get_current_ops()
    set_current_ops(get_ops(name, **kwargs))
    yield
    set_current_ops(current_ops)


def get_current_ops() -> Ops:
    return context_ops.get()


def set_current_ops(ops: Ops) -> None:
    context_ops.set(ops)
    _get_thread_state().ops = ops


def contextvars_eq_thread_ops() -> bool:
    current_ops = context_ops.get()
    thread_ops = _get_thread_state().ops
    if type(current_ops) == type(thread_ops):
        return True
    return False


def _get_thread_state():
    thread: threading.Thread = threading.current_thread()
    if not hasattr(thread, "__local"):
        thread.__local = _create_thread_local(_GLOBAL_STATE)
    return thread.__local


def _create_thread_local(
    attrs: Dict[str, Any], local_class: Type[threading.local] = threading.local
):
    obj = local_class()
    for name, value in attrs.items():
        setattr(obj, name, value)
    return obj


__all__ = [
    "set_current_ops",
    "get_current_ops",
    "use_ops",
    "ParamServer",
    "Ops",
    "CupyOps",
    "NumpyOps",
    "has_cupy",
]
