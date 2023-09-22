#!/usr/bin/env python3

"""Test dist_utils.py."""

import argparse
from functools import partial
import os

import torch
import torch.distributed as dist

import dist_utils as du

printf0 = partial(du.print0, flush=True)


def assert_equal(actual, expected):
    """Assert that two objects are equal."""
    assert actual == expected, f"{actual} != {expected}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-type", required=True, choices=["cpu", "cuda"])
    args = parser.parse_args()

    du.init_distributed()
    printf0("Testing init_distributed...")

    rank = dist.get_rank()
    local_rank = dist.get_rank(du.get_local_group())
    group_rank = int(os.environ.get("GROUP_RANK", "0"))
    world_size = dist.get_world_size()
    printf0(f"World size: {world_size}")

    if dist.is_torchelastic_launched():
        printf0("Testing local rank...")
        assert_equal(local_rank, int(os.environ["LOCAL_RANK"]))

    printf0("Testing do_in_order (please verify that these print in order)...")
    with du.do_in_order():
        print(f"Hello from rank {rank}")

    printf0("Testing do_in_local_order (please verify that these print in local order)...")
    with du.do_in_local_order():
        print(f"Hello from group rank {group_rank}, local rank {local_rank}")

    printf0("Testing printl0 (please verify only local rank 0 prints)...")
    du.printl0(f"Hello from rank {rank}, local rank {local_rank}", flush=True)

    printf0("Testing get_device...")
    device = du.get_device() if args.device_type == "cuda" else torch.device("cpu")
    if args.device_type == "cuda":
        assert_equal(device.index, local_rank)

    torch.set_default_device(device)

    printf0("Testing all_gather_object...")
    actual = du.all_gather_object(rank)
    expected = list(range(world_size))
    assert_equal(actual, expected)

    printf0("Testing broadcast_object...")
    actual = du.broadcast_object(rank)
    expected = 0
    assert_equal(actual, expected)

    printf0("Testing gather_object...")
    actual = du.gather_object(rank)
    expected = list(range(world_size)) if rank == 0 else None
    assert_equal(actual, expected)

    printf0("Testing scatter_objects...")
    actual = du.scatter_objects(list(range(world_size)))
    expected = rank
    assert_equal(actual, expected)

    printf0("Testing all_gather_into_new...")
    tensor = torch.eye(rank + 1)
    actual = du.all_gather_into_new(tensor)
    expected = [torch.eye(i) for i in range(1, world_size + 1)]
    for a, e in zip(actual, expected):
        assert torch.all(a == e), f"{a} != {e}"

    printf0("Testing broadcast_tensors...")
    torch.manual_seed(rank)
    actual = [torch.randn(4, 4), torch.randn(6, 6)]
    du.broadcast_tensors(actual)
    torch.manual_seed(0)
    expected = [torch.randn(4, 4), torch.randn(6, 6)]
    for a, e in zip(actual, expected):
        assert torch.all(a == e), f"{a} != {e}"

    printf0("Testing cleanup_distributed...")
    du.cleanup_distributed()
    assert_equal(du.get_local_group(), None)
    success = False
    try:
        dist.barrier()
    except ValueError:
        success = True
    assert success, "dist.barrier() should fail after cleanup_distributed()"


if __name__ == "__main__":
    main()
