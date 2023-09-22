"""Utilities for PyTorch distributed."""

import atexit
from collections import defaultdict
from contextlib import contextmanager
from functools import wraps
import os
from typing import Any, Callable, Iterable, List, Optional

import torch
import torch.distributed as dist


# Type hints

Group = Optional[dist.ProcessGroup]


# State

all_groups = []
local_group = None


def get_local_group() -> dist.ProcessGroup:
    """Get the process group containing only the local processes.

    :returns: The process group containing only the local processes.
    """
    return local_group


def get_device() -> torch.device:
    """Get the device of the current process.

    :returns: The device of the current process.
    """
    if torch.cuda.is_available():
        return torch.device("cuda", dist.get_rank(local_group))
    return torch.device("cpu")


# Initialization and cleanup


def init_distributed():
    """Initialize distributed communication. If the process is not launched with ``torchrun``,
    then assume it is the only process."""
    global local_group
    backend = None if torch.cuda.is_available() else "gloo"
    if dist.is_torchelastic_launched():
        dist.init_process_group(backend)
        all_groups.append(None)
        group_rank = int(os.environ["GROUP_RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        rank_list = all_gather_object((group_rank, local_rank))
        groups = defaultdict(list)
        for global_rank, (group_rank, local_rank) in enumerate(rank_list):
            groups[group_rank].append((local_rank, global_rank))
        ranks_per_group = [[rank[1] for rank in sorted(group)] for group in groups.values()]
        local_group, new_groups = dist.new_subgroups_by_enumeration(ranks_per_group)
        all_groups.extend(new_groups)
        if torch.cuda.is_available():
            torch.cuda.set_device(get_device())
    else:
        dist.init_process_group(backend, world_size=1, rank=0, store=dist.HashStore())
        all_groups.append(None)
        local_group = dist.new_group([0])
        all_groups.append(local_group)
    atexit.register(cleanup_distributed)


def cleanup_distributed():
    """Clean up distributed communication."""
    global local_group
    atexit.unregister(cleanup_distributed)
    for group in reversed(all_groups):
        dist.destroy_process_group(group)
    all_groups.clear()
    local_group = None


# Context managers and decorators


@contextmanager
def do_in_order(group: Group = None):
    """A context manager that ensures that all processes execute the block in order.

    :param group: The process group. If ``None``, use the default group.
    """
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    dist.barrier(group)
    for i in range(world_size):
        if rank == i:
            yield
        dist.barrier(group)


@contextmanager
def do_in_local_order():
    """A context manager that ensures that all local processes execute the block in order."""
    with do_in_order(local_group):
        yield


def on_rank_0(group: Group = None) -> Callable:
    """A decorator that ensures that only process 0 executes the function.

    :param group: The process group. If ``None``, use the default group.
    :returns: The decorated function.
    """

    def decorator(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            if dist.get_rank(group) == 0:
                return fn(*args, **kwargs)

        return wrapped

    return decorator


def on_local_rank_0() -> Callable:
    """A decorator that ensures that only the local process 0 executes the function.

    :returns: The decorated function.
    """

    def decorator(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            if dist.get_rank(local_group) == 0:
                return fn(*args, **kwargs)

        return wrapped

    return decorator


print0 = on_rank_0()(print)
print0.__doc__ = """A version of ``print()`` that only prints on process 0."""
printl0 = on_local_rank_0()(print)
printl0.__doc__ = """A version of ``print()`` that only prints on local process 0."""

# Transferring objects


def all_gather_object(obj: Any, group: Group = None) -> List[Any]:
    """Gather an object from each process and return a list of gathered objects in all
    processes.

    :param obj: The object to gather.
    :param group: The process group. If ``None``, use the default group.
    :returns: A list of gathered objects.
    """
    object_list = [None] * dist.get_world_size(group)
    dist.all_gather_object(object_list, obj, group=group)
    return object_list


def broadcast_object(obj: Optional[Any] = None, src: int = 0, group: Group = None) -> Any:
    """Broadcast an object from the source process and return the object in all processes.

    :param obj: The object to broadcast. Ignored in processes other than ``src``.
    :param src: The source process.
    :param group: The process group. If ``None``, use the default group.
    :returns: The object broadcasted from the source process.
    """
    object_list = [obj]
    dist.broadcast_object_list(object_list, src=src, group=group)
    return object_list[0]


def gather_object(obj: Any, dst: int = 0, group: Group = None) -> Optional[List[Any]]:
    """Gather an object from each process and return a list of gathered objects in the
    destination process.

    :param obj: The object to gather.
    :param dst: The destination process.
    :param group: The process group. If ``None``, use the default group.
    :returns: A list of gathered objects in the destination process, ``None`` in all other
        processes.
    """
    rank = dist.get_rank(group)
    object_list = [None] * dist.get_world_size(group) if rank == dst else None
    dist.gather_object(obj, object_list, dst=dst, group=group)
    if rank == dst:
        return object_list


def scatter_objects(objs: Optional[List[Any]] = None, src: int = 0, group: Group = None) -> Any:
    """Scatter a list of objects from the source process and return each object in each
    process.

    :param objs: The list of objects to scatter. Ignored in processes other than ``src``.
    :param src: The source process.
    :param group: The process group. If ``None``, use the default group.
    :returns: The object scattered to the current process.
    """
    object_list = [None]
    dist.scatter_object_list(object_list, objs, src=src, group=group)
    return object_list[0]


# Transferring tensors


def all_gather_into_new(tensor: torch.Tensor, group: Group = None) -> List[torch.Tensor]:
    """Gather a tensor from each process and return a list of gathered tensors in all
    processes. Tensors can have different shapes. Tensors must be all on CPU or all on GPU.

    :param tensor: The tensor to gather.
    :param group: The process group. If ``None``, use the default group.
    :returns: A list of gathered tensors.
    """
    shapes = all_gather_object(tensor.shape, group=group)
    if tensor.device.type == "cuda":
        tensors = [torch.empty(shape, dtype=tensor.dtype, device=tensor.device) for shape in shapes]
        dist.all_gather(tensors, tensor, group=group)
    else:
        rank = dist.get_rank(group)
        tensors = []
        for i, shape in enumerate(shapes):
            if i == rank:
                tensors.append(tensor)
            else:
                tensors.append(torch.empty(shape, dtype=tensor.dtype, device=tensor.device))
        handles = [
            dist.broadcast(tensors[i], src=i, group=group, async_op=True)
            for i in range(len(tensors))
        ]
        for handle in handles:
            handle.wait()
    return tensors


def broadcast_tensors(tensors: Iterable[torch.Tensor], src: int = 0, group: Group = None):
    """Broadcast an iterable of tensors from the given source process to all other processes.

    To synchronize a model's parameters in all processes to the versions in process 0:

    .. code-block:: python

        broadcast_tensors(model.parameters())

    :param tensors: The tensors to broadcast.
    :param src: The source process.
    :param group: The process group. If ``None``, use the default group.
    """
    handles = [dist.broadcast(tensor, src=src, group=group, async_op=True) for tensor in tensors]
    for handle in handles:
        handle.wait()
