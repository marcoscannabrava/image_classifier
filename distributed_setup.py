import os
import torch

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
from time import monotonic

import lightning as L


# Multi-process setup/cleanup functions
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def prep_data(dataset, world_size, rank, batch_size, num_workers=0):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, drop_last=True)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler) # type: ignore

def distributed_training(rank, world_size, model, train_fn, dataset, hyperparameters, fabric, tb):
    start = monotonic()
    if rank is not None:
        print("distributed_training on:", rank)    
        setup(rank, world_size)

    criterion, optimizer = model.setup_optimizer()
    model, optimizer = fabric.setup(model, optimizer)
    dataloader = DataLoader(dataset)
    dataloader = fabric.setup_dataloaders(dataloader)
    
    ddp_model = DDP(model, device_ids=[rank]) if not fabric else model
    loss = train_fn(rank, ddp_model, optimizer, criterion, dataloader, fabric=fabric)
    # DDP automatically syncs gradients after backward() pass so we know that,
    # after training, process 0 will have all gradients to save the model
    if rank == 0 or rank is None:
        PATH = './cifar_net_gpu_distributed.pth'
        torch.save(model.state_dict(), PATH)
        tb.add_hparams(
            {"batch_size": hyperparameters.get("batch_size"), "num_workers": hyperparameters.get("num_workers")},
            { "loss": loss, "runtime": monotonic() - start }
        )
        tb.close()
    if rank is not None:
        cleanup()

def run(model, train_fn, dataset, hyperparameters):
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
    
    fabric = L.Fabric(accelerator='cuda' if 'cuda' in device else 'cpu', devices=world_size, strategy="ddp")
    fabric.launch()

    tb = SummaryWriter(comment=f'batch_size={hyperparameters.get("batch_size")} num_workers={hyperparameters.get("num_workers")}')
    start = monotonic()

    mp.spawn(
        distributed_training,
        args=(world_size, model, train_fn, dataset, hyperparameters, fabric, tb),
        nprocs=world_size,
        join=True
    )

