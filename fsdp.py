import lightning as L
import torch

from model_lightning import net
from data import trainloader, testloader, classes

trainer = L.Trainer(accelerator='gpu', max_epochs=2, strategy='fsdp', enable_checkpointing=True)

PATH = './cifar_fsdp.pth'

def main():
    trainer.fit(model=net, train_dataloaders=trainloader, val_dataloaders=testloader)
    torch.save(net.state_dict(), PATH)

main()