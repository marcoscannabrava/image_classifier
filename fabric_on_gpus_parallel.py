import lightning as L

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import Net
from data import *


model = Net()

fabric = L.Fabric(accelerator='cuda', devices=4, strategy="ddp")
fabric.launch()

criterion, optimizer = nn.CrossEntropyLoss(), optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

model, optimizer = fabric.setup(model, optimizer)

dataloader = fabric.setup_dataloaders(DataLoader(trainset))


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
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

PATH = './cifar_net_gpu_distributed.pth'
torch.save(model.state_dict(), PATH)
