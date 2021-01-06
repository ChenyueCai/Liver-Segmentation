import torch
import torch.utils.data
import torch.nn as nn
import torchvision
import torchvision.models
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
import time
import os
from segmentation.toy_dataloader import *

# In the future can defined
class ToySegNet(nn.Module):
    def __init__(self):
        super().__init__()


def train_test_split(train_prop, val_prop, data_set):
    train_size, val_size = int(len(data_set) * train_prop), int(len(data_set) * val_prop)
    test_size = len(data_set) - train_size - val_size
    assert test_size > 0
    train_set, test_set = torch.utils.data.random_split(data_set, [train_size + val_size, test_size])
    train_set, val_set = torch.utils.data.random_split(train_set, [train_size, val_size])
    return train_set, val_set, test_set


def pixel_error(prediction, truth, h, w):
    return torch.sum(torch.flatten(prediction) == torch.flatten(truth)).item() / (h * w)


def train(model, epoch, train_loader, val_loader, optimizer, criterion, device):
    train_error, val_error = [], []
    best_model = copy.deepcopy(model.state_dict())

    for ep in range(1, epoch+1):

        model.train()
        train_step_error = 0
        for step in range(1, len(train_loader)+1):
            images, seg = next(iter(train_loader))
            images, seg = images.to(device), seg.to(device)
            predictions = model(images)['out']
            print(predictions.shape, seg.shape)
            optimizer.zero_grad()
            loss_train_step = criterion(predictions, seg.long())
            loss_train_step.backward()
            optimizer.step()
            train_step_error += loss_train_step.item()
        train_error.append(train_step_error/len(train_loader))
        print('done training with ep' + ep)

        model.eval()
        val_step_error = 0
        with torch.no_grad():
            for step in range(1, len(val_loader)+1):
                images, seg = next(iter(val_loader))
                images, seg = images.to(device), seg.to(device)
                predictions = model(images)['out']
                loss_valid_step = criterion(predictions, seg.long())
                val_step_error += loss_valid_step.item()
        val_error.append(val_step_error/len(val_loader))

        print('done evaluating with ep' + ep)

        if val_error[-1] <= min(val_error):
            best_model = copy.deepcopy(model.state_dict())

    return best_model, train_error, val_error


def test(test_loader, device, model, criterion):
    error = 0
    predicted_segment = []
    pix_error = []
    with torch.no_grad():
        for step in range(1, len(test_loader)+1):
            images, seg = next(iter(test_loader))
            images, seg = images.to(device), seg.to(device)
            prediction = model(images)['out']
            loss_valid_step = criterion(prediction, seg.long())
            error += loss_valid_step.item()

            _, pred = torch.max(prediction, 1)
            predicted_segment.append(pred)

            pix_error.append(pixel_error(pred, seg, 512, 512))

    return error/len(test_loader), predicted_segment, pix_error


def main():

    data_set = LiverSegSet(ct_dir='ct_dirs', seg_dir='seg_files', h=512, w=512)
    train_set, val_set, test_set = train_test_split(0.7, 0.1, data_set)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=4, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=False)

    print('starting')
    epoch = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.segmentation.fcn_resnet101(pretrained=False, progress=True, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    print('start train')
    best_model, train_error, val_error = train(model, epoch, train_loader, val_loader, optimizer, criterion, device)

    print(train_error)
    print(val_error)


if __name__ == '__main__':
    main()
