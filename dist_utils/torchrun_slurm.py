#!/usr/bin/env python3

"""Launch a PyTorch distributed job on a Slurm cluster."""

import os
from subprocess import Popen, PIPE
import sys


def main():
    proc = Popen(["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]], stdout=PIPE)
    master_addr = proc.stdout.read().decode().splitlines()[0]

    command = [
        "torchrun",
        "--nnodes",
        os.environ["SLURM_NPROCS"],
        "--nproc-per-node",
        "gpu",
        "--node_rank",
        os.environ["SLURM_PROCID"],
        "--master_addr",
        master_addr,
        *sys.argv[1:],
    ]
    Popen(command).wait()


if __name__ == "__main__":
    main()
