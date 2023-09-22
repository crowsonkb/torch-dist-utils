# torch-dist-utils

Utilities for PyTorch distributed.

[API documentation](https://crowsonkb.github.io/torch-dist-utils/)

## Example

There is an example script to train a classifier on CIFAR-10 using DDP in `examples/train_classifier.py`. It can be run on CPU or one GPU with:

```sh
python examples/train_classifier.py
```

and on multiple GPUs with:

```sh
torchrun --nproc-per-node gpu examples/train_classifier.py
```

## Slurm wrapper for torchrun

`torchrun_slurm` is a wrapper for `torchrun` that can be used with Slurm. It is a drop-in replacement for `torchrun`. It will automatically set the `--nnodes`, `--node-rank`, and `--master-addr` arguments from `$SLURM_NPROCS`, `$SLURM_PROCID`, and the first node in `$SLURM_JOB_NODELIST` respectively. It also sets `--nproc-per-node` to the number of GPUs on the node (you can override it by setting it explicitly).

## Test cases

Test cases should run on CPU and GPU, with and without torchrun.

```sh
python -m torch_dist_utils.test --device-type cpu

python -m torch_dist_utils.test --device-type cuda

CUDA_VISIBLE_DEVICES="" torchrun --nproc-per-node 4 -m torch_dist_utils.test --device-type cpu

torchrun --nproc-per-node gpu -m torch_dist_utils.test --device-type cuda
```

Test cases should also run on multiple nodes. To simulate this on a single machine, run:

```sh
CUDA_VISIBLE_DEVICES="" torchrun --master-addr localhost --master-port 25500 --nnodes 2 --nproc-per-node 4 --node-rank 0 -m torch_dist_utils.test --device-type cpu
```

in one terminal, and

```sh
CUDA_VISIBLE_DEVICES="" torchrun --master-addr localhost --master-port 25500 --nnodes 2 --nproc-per-node 4 --node-rank 1 -m torch_dist_utils.test --device-type cpu
```

in another.
