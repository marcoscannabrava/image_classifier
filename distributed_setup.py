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
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def prep_data(dataset, batch_size, world_size, rank, num_workers=0):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, drop_last=True)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler) # type: ignore

def distributed_training(rank, world_size, model, train_fn, dataset, batch_size, num_workers):
    print("distributed_training on:", rank)
    setup(rank, world_size)
    model = model.to(rank)
    sampled_dataloader = prep_data(dataset, batch_size, world_size, rank, num_workers)
    criterion, optimizer = model.setup_optimizer()
    ddp_model = DDP(model, device_ids=[rank])
    train_fn(rank, ddp_model, optimizer, criterion, sampled_dataloader)

    # DDP automatically syncs gradients after backward() pass so we know that,
    # after training, process 0 will have all gradients to save the model
    if rank == 0:
        PATH = './cifar_net_gpu_distributed.pth'
        torch.save(model.state_dict(), PATH)
    cleanup()

def run(model, train_fn, dataset, batch_size, num_workers):
    device = 'cpu'
    if torch.backends.mps.is_available(): # type: ignore
        device = 'mps'  # enables training on the Macbook Pro's GPU
    elif torch.cuda.is_available():
        device = 'cuda:0'

    print("Running on ", torch.cuda.device_count() or 1, device)

    if device == 'cpu':
        raise Exception("No GPU found.")
    
    if device == 'mps':
        raise Exception("Mac has a single GPU.")

    world_size = torch.cuda.device_count()

    mp.spawn(
        distributed_training,
        args=(world_size, model, train_fn, dataset, batch_size, num_workers),
        nprocs=world_size,
        join=True
    )
