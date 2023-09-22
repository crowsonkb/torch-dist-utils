# dist-utils

Utilities for PyTorch distributed.

[Documentation](https://crowsonkb.github.io/dist-utils/)

## Test cases

Test cases should run on CPU and GPU, with and without torchrun.

```sh
python -m dist_utils.test --device-type cpu

python -m dist_utils.test --device-type cuda

CUDA_VISIBLE_DEVICES="" torchrun --nproc-per-node 4 -m dist_utils.test --device-type cpu

torchrun --nproc-per-node gpu -m dist_utils.test --device-type cuda
```

Test cases should also run on multiple nodes. To simulate this on a single machine, run:

```sh
CUDA_VISIBLE_DEVICES="" torchrun --master-addr localhost --master-port 25500 --nnodes 2 --nproc-per-node 4 --node-rank 0 -m dist_utils.test --device-type cpu
```

in one terminal, and

```sh
CUDA_VISIBLE_DEVICES="" torchrun --master-addr localhost --master-port 25500 --nnodes 2 --nproc-per-node 4 --node-rank 1 -m dist_utils.test --device-type cpu
```

in another.

## Slurm wrapper for torchrun

`torchrun_slurm` is a wrapper for torchrun that can be used with Slurm. It is a drop-in replacement for torchrun. It will automatically set the `--nnodes`, `--node-rank`, and `--master-addr` arguments from `$SLURM_NPROCS`, `$SLURM_PROCID`, and the first node in `$SLURM_JOB_NODELIST` respectively. It also sets `--nproc-per-node` to the number of GPUs on the node (you can override it by setting it explicitly).
