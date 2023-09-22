"""Utilities for PyTorch distributed."""

import atexit
from collections import defaultdict
from contextlib import contextmanager
from functools import wraps
import os

import torch
import torch.distributed as dist


# State

all_groups = []
local_group = None


def get_local_group():
    """Get the group containing only the local processes."""
    return local_group


def get_device():
    """Get the device of the current process."""
    if torch.cuda.is_available():
        return torch.device("cuda", dist.get_rank(local_group))
    return torch.device("cpu")


# Initialization and cleanup


def init_distributed():
    """Initialize distributed communication. If the process is not launched with
    torchrun/torchelastic, then assume it is the only process."""
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
    """Cleanup distributed communication."""
    global local_group
    atexit.unregister(cleanup_distributed)
    for group in reversed(all_groups):
        dist.destroy_process_group(group)
    all_groups.clear()
    local_group = None


# Context managers and decorators


@contextmanager
def do_in_order(group=None):
    """A context manager that ensures that all processes execute the block in order."""
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


def on_rank_0(group=None):
    """A decorator that ensures that only process 0 executes the function."""

    def decorator(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            if dist.get_rank(group) == 0:
                return fn(*args, **kwargs)

        return wrapped

    return decorator


def on_local_rank_0():
    """A decorator that ensures that only the local process 0 executes the function."""

    def decorator(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            if dist.get_rank(local_group) == 0:
                return fn(*args, **kwargs)

        return wrapped

    return decorator


print0 = on_rank_0()(print)
printl0 = on_local_rank_0()(print)


# Transferring objects


def all_gather_object(obj, group=None):
    """Gather an object from each process and return a list of gathered objects in all
    processes."""
    object_list = [None] * dist.get_world_size(group)
    dist.all_gather_object(object_list, obj, group=group)
    return object_list


def broadcast_object(obj=None, src=0, group=None):
    """Broadcast an object from the source process and return the object in all processes."""
    object_list = [obj]
    dist.broadcast_object_list(object_list, src=src, group=group)
    return object_list[0]


def gather_object(obj, dst=0, group=None):
    """Gather an object from each process and return a list of gathered objects in the
    destination process."""
    rank = dist.get_rank(group)
    object_list = [None] * dist.get_world_size(group) if rank == dst else None
    dist.gather_object(obj, object_list, dst=dst, group=group)
    if rank == dst:
        return object_list


def scatter_objects(objs=None, src=0, group=None):
    """Scatter a list of objects from the source process and return each object in each
    process."""
    object_list = [None]
    dist.scatter_object_list(object_list, objs, src=src, group=group)
    return object_list[0]


# Transferring tensors


def all_gather_into_new(tensor, group=None):
    """Gather a tensor from each process and return a list of gathered tensors in all
    processes. Tensors can have different shapes. Tensors must be all on CPU or all on GPU.
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


def broadcast_tensors(tensors, src=0, group=None):
    """Broadcast an iterable of tensors from the given source process to all other processes.

    For instance, `broadcast_tensors(model.parameters())` will synchronize the model
    parameters in all processes to the versions on rank 0."""
    handles = [dist.broadcast(tensor, src=src, group=group, async_op=True) for tensor in tensors]
    for handle in handles:
        handle.wait()
