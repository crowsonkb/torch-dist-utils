#!/usr/bin/env python3

"""Trains a classifier on CIFAR-10 using torch_dist_utils and DDP.

It can be run on CPU or one GPU with:

    python train_classifier.py

or on multiple GPUs with:

    torchrun --nproc-per-node gpu train_classifier.py
"""

import argparse
from contextlib import contextmanager, nullcontext

import torch
from torch import distributed as dist, nn, optim
from torch.nn import functional as F
from torch.utils import data
import torch_dist_utils as du
from torchvision import datasets, transforms
from tqdm import tqdm

print0 = tqdm.external_write_mode()(du.print0)


@contextmanager
def train_mode(model, mode=True):
    """A context manager that places a model into training mode and restores
    the previous mode on exit."""
    modes = [module.training for module in model.modules()]
    try:
        model.train(mode)
        yield
    finally:
        for i, module in enumerate(model.modules()):
            module.training = modes[i]


def eval_mode(model):
    """A context manager that places a model into evaluation mode and restores
    the previous mode on exit."""
    return train_mode(model, False)


def sync_grads(model, enable=True):
    """A context manager that enables or disables gradient synchronization."""
    return nullcontext() if enable else model.no_sync()


@torch.no_grad()
def gradient_norm(params):
    """Computes the 2-norm of the gradients of the given parameters."""
    total = torch.tensor(0.0)
    for p in params:
        if p.grad is not None:
            total = total.to(p.grad.device)
            total += torch.norm(p.grad, dtype=total.dtype) ** 2
    return torch.sqrt(total)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    du.init_distributed()

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = du.get_device()

    print0(f"Batch size per process: {args.batch_size}")
    print0(f"World size: {world_size}")
    print0(f"Batch size: {args.batch_size * world_size}")

    if rank == 0:
        datasets.CIFAR10("_data", download=True)
    dist.barrier()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),
        ]
    )

    train_set = datasets.CIFAR10("_data", train=True, transform=transform)
    train_sampler = data.DistributedSampler(
        train_set, num_replicas=world_size, rank=rank, seed=args.seed
    )
    train_dl = data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=4,
        persistent_workers=True,
        drop_last=True,
    )

    val_set = datasets.CIFAR10("_data", train=False, transform=transform)
    val_sampler = data.DistributedSampler(
        val_set, num_replicas=world_size, rank=rank, seed=args.seed, shuffle=False
    )
    val_dl = data.DataLoader(
        val_set,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=4,
        persistent_workers=True,
    )

    torch.manual_seed(args.seed)
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.GELU(),
        nn.Dropout2d(0.25),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.GELU(),
        nn.Dropout2d(0.25),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * 8 * 8, 64),
        nn.GELU(),
        nn.Dropout(0.25),
        nn.Linear(64, 10),
    ).to(device)
    du.broadcast_tensors(model.parameters())

    print0(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    dist_kwargs = {}
    if device.type == "cuda":
        dist_kwargs["device_ids"] = [device.index]
        dist_kwargs["output_device"] = device.index
    model = nn.parallel.DistributedDataParallel(model, **dist_kwargs)

    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-1)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, verbose=rank == 0)

    @torch.no_grad()
    @eval_mode(model)
    def validate():
        print0("Validating...")
        correct = torch.zeros((), dtype=torch.long, device=device)
        total = torch.zeros((), dtype=torch.long, device=device)
        for x, y in tqdm(val_dl, disable=rank != 0):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            correct_batch = (logits.argmax(dim=-1) == y).sum()
            total_batch = torch.tensor(y.numel(), device=device)
            dist.all_reduce_coalesced([correct_batch, total_batch])
            correct += correct_batch
            total += total_batch
        acc = correct / total
        print0(f"Validation accuracy: {acc:.2%}")

    step = 0
    substep = 1
    loss_parts = []

    try:
        for epoch in range(args.epochs):
            train_sampler.set_epoch(epoch)
            for x, y in tqdm(train_dl, disable=rank != 0):
                sync = substep >= args.grad_accum_steps
                x, y = x.to(device), y.to(device)

                with sync_grads(model, sync):
                    logits = model(x)
                    loss = F.cross_entropy(logits, y)

                    loss_part = loss / args.grad_accum_steps
                    loss_part.backward()
                    loss_parts.append(loss_part.detach())
                if sync:
                    opt.step()
                    substep = 0

                    if step % 25 == 0:
                        loss_global = sum(loss_parts) / world_size
                        grad_norm = gradient_norm(model.parameters())
                        dist.all_reduce(loss_global)
                        print0(
                            f"epoch: {epoch}, step: {step}, loss: {loss_global.item():g}, grad norm: {grad_norm.item():g}"
                        )

                    step += 1
                    loss_parts.clear()
                    opt.zero_grad()

                substep += 1

            validate()
            sched.step()

        print0("Done!")
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
