#!/usr/bin/env python3

"""Launch a PyTorch distributed job on a Slurm cluster."""

import os
from subprocess import Popen, PIPE
import sys


def main():
    try:
        nnodes = os.environ["SLURM_NPROCS"]
        node_rank = os.environ["SLURM_PROCID"]
        nodelist = os.environ["SLURM_JOB_NODELIST"]
    except KeyError:
        print(
            "This script must be run within a Slurm job. The environment variables "
            "SLURM_NPROCS, SLURM_PROCID, and SLURM_JOB_NODELIST were not found.",
            file=sys.stderr,
        )
        sys.exit(1)

    proc = Popen(["scontrol", "show", "hostnames", nodelist], stdout=PIPE)
    master_addr = proc.stdout.read().decode().splitlines()[0]

    command = [
        "torchrun",
        "--nnodes",
        nnodes,
        "--nproc-per-node",
        "gpu",
        "--node_rank",
        node_rank,
        "--master_addr",
        master_addr,
        *sys.argv[1:],
    ]
    Popen(command).wait()


if __name__ == "__main__":
    main()
