from time import monotonic
from itertools import product

import lightning as L
from lightning.fabric.loggers import TensorBoardLogger

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import Net
from data import *

HP_BATCH_SIZE = [100,5000]
HP_NUM_WORKERS = [0,4,16]

experiments = product(HP_BATCH_SIZE, HP_NUM_WORKERS)

for batch_size, num_workers in experiments:
    print(f"Running training... batch_size={batch_size}, num_workers={num_workers}")
    model = Net()

    logger = TensorBoardLogger(root_dir="logs")
    fabric = L.Fabric(accelerator='cuda', devices=4, strategy="ddp", loggers=logger)
    fabric.launch()
    criterion, optimizer = nn.CrossEntropyLoss(), optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model, optimizer = fabric.setup(model, optimizer)
    dataloader = fabric.setup_dataloaders(DataLoader(trainset, batch_size=batch_size, num_workers=num_workers))

    logger.log_hyperparams({ 'batch_size': batch_size, 'num_workers': num_workers })

    start = monotonic()
    running_loss = 0.0
    for epoch in range(2): 

        for i, data in enumerate(dataloader, 0):
            inputs, labels = data

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            fabric.backward(loss)
            optimizer.step()

            running_loss += loss.item()

    PATH = f'./experiments/fabric-batch_size={batch_size}-num_workers={num_workers}.pth'
    fabric.save(PATH, { 'model': model.state_dict() })
    
    runtime = monotonic() - start
    print(f"Finished training. loss={running_loss}, runtime={runtime}")
    fabric.log_dict({ 'loss': running_loss, 'runtime': runtime })
