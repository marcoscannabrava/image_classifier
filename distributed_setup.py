import os
import torch

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader



# Multi-process setup/cleanup functions
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size) # or "gloo"

def cleanup():
    dist.destroy_process_group()

def prep_data(dataset, batch_size, world_size, rank):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, sampler=sampler) # type: ignore

def distributed_training(rank, world_size, model, train_fn, dataset, batch_size):
    print("distributed_training on:", rank)
    setup(rank, world_size)
    model = model.to(rank)
    sampled_dataloader = prep_data(dataset, batch_size, world_size, rank)
    criterion, optimizer = model.setup_optimizer()
    ddp_model = DDP(model, device_ids=[rank])
    train_fn(rank, ddp_model, optimizer, criterion, sampled_dataloader)
    cleanup()

def run(model, train_fn, dataset, batch_size):
    device = 'cpu'
    if torch.backends.mps.is_available(): # type: ignore
        device = 'mps'  # enables training on the Macbook Pro's GPU
    elif torch.cuda.is_available():
        device = 'cuda:0'

    print("Running on ", torch.cuda.device_count(), device)

    if device == 'cpu':
        raise Exception("No GPU found.")

    world_size = torch.cuda.device_count()

    mp.spawn(
        distributed_training,
        args=(world_size, model, train_fn, dataset, batch_size),
        nprocs=world_size,
        join=True
    )