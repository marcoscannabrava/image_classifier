import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data import trainloader, testloader, classes


# Model definition

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def setup_optimizer(self):
        """ Returns loss function and sets up optimizer"""
        return nn.CrossEntropyLoss(), optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

net = Net()
default_criterion, default_optimizer = net.setup_optimizer()

# Training
def train(device=None, model=net, optimizer=default_optimizer, loss_fn=default_criterion, dataloader=trainloader):
    running_loss = 0.0
    for epoch in range(2):  # loop over the dataset multiple times

        if isinstance(device, int): # HACK: if device is int, then we're using DDP
            dataloader.sampler.set_epoch(epoch) # type: ignore

        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = (data[0].to(device), data[1].to(device)) if device != 'cpu' else data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
    
    print('Finished Training')
    return running_loss
        


def test(model, device=None):
    correct = 0
    total = 0

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = (data[0].to(device), data[1].to(device)) if device else data
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for label, prediction in zip(labels, predicted):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')