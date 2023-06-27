import os
import torch

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
from time import monotonic


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

def distributed_training(rank, world_size, model, train_fn, dataset, hyperparameters, tb):
    print("distributed_training on:", rank)
    start = monotonic()
    setup(rank, world_size)
    model = model.to(rank)
    sampled_dataloader = prep_data(
        dataset,
        world_size,
        rank,
        batch_size=hyperparameters.get('batch_size'),
        num_workers=hyperparameters.get('num_workers')
    )
    criterion, optimizer = model.setup_optimizer()
    ddp_model = DDP(model, device_ids=[rank])
    loss = train_fn(rank, ddp_model, optimizer, criterion, sampled_dataloader)
    # DDP automatically syncs gradients after backward() pass so we know that,
    # after training, process 0 will have all gradients to save the model
    if rank == 0:
        PATH = './cifar_net_gpu_distributed.pth'
        torch.save(model.state_dict(), PATH)
        tb.add_hparams(
            {"batch_size": hyperparameters.get("batch_size"), "num_workers": hyperparameters.get("num_workers")},
            { "loss": loss, "runtime": monotonic() - start }
        )
        tb.close()
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

    tb = SummaryWriter(comment=f'batch_size={hyperparameters.get("batch_size")} num_workers={hyperparameters.get("num_workers")}')

    mp.spawn(
        distributed_training,
        args=(world_size, model, train_fn, dataset, hyperparameters, tb),
        nprocs=world_size,
        join=True
    )

