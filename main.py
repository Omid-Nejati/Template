import os
import sys
import numpy as np
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

import torchvision.utils
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torchsummary import summary

from tqdm import tqdm
import medmnist
from medmnist import INFO, Evaluator
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import natten
import torchattacks
from torchattacks import PGD, FGSM

from MedViT import MedViT_tiny, MedViT_small, MedViT_base, MedViT_large




def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    model_name = 'MedViT_tiny'
   

    data_flag = 'breastmnist'
# [tissuemnist, pathmnist, chestmnist, dermamnist, octmnist,
# pnemoniamnist, retinamnist, breastmnist, bloodmnist, tissuemnist, organamnist, organcmnist, organsmnist]

    # dataset 
    download = True

    batch_size = 32
    lr = 0.007

    info = INFO[data_flag]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])

    DataClass = getattr(medmnist, info['python_class'])

    print("number of channels : ", n_channels)
    print("number of classes : ", n_classes)
    
    net = MedViT_tiny(num_classes=n_classes)
    net.to(device)
    
    path = '/content/drive/MyDrive/MedViTV2_tiny.pth'
    checkpoint = torch.load(path)
    state_dict = net.state_dict()
    checkpoint_model = checkpoint['model']
    for k in ['proj_head.0.weight', 'proj_head.0.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]
    net.load_state_dict(checkpoint_model, strict=False)
    
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.007, momentum=0.9)
    
    from torchvision.transforms.transforms import Resize
    # preprocessing
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.Lambda(lambda image: image.convert('RGB')),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.Lambda(lambda image: image.convert('RGB')),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    # load the data
    train_dataset = DataClass(split='train', transform=train_transform, download=download)
    test_dataset = DataClass(split='test', transform=test_transform, download=download)

    val_num = len(test_dataset)
    # pil_dataset = DataClass(split='train', download=download)

    # encapsulate data into dataloader form
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*batch_size, shuffle=False)
    
    print(train_dataset)
    print("===================")
    print(test_dataset)

    epochs = 100
    best_acc = 0.0
    save_path = './{}.pth'.format(model_name)
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, datax in enumerate(train_bar):
            images, labels = datax
            optimizer.zero_grad()
            outputs = net(images.to(device))
            labels = labels.squeeze().long()
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(test_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            print('Saving checkpoint...')
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
            print()

    print('Finished Training')


if __name__ == '__main__':
    main()
