import lightning as L
import torch
import torchvision


from model_lightning import LitNet
from data import testloader, classes


PATH = '/project/jobs/double-4t4-attempt-4/nodes.0/cifar_fsdp.pth'

def main():
    net = LitNet()
    net.load_state_dict(torch.load(PATH))
    net.eval()
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    dataiter = iter(testloader)
    images, labels = next(dataiter)

    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(4)))

main()