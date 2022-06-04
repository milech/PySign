# -----------------------------------------------------------
# Handwritten signature verification system using DTW and CNN
#
#
# (C) 2022 Michal Lech, Gdynia, Poland
# Released under GNU General Public License v3.0 (GPL-3.0)
# email: mlech.ksm@gmail.com
# -----------------------------------------------------------

from typing import Any, Tuple
import torch
import torch.nn as nn
import torch.utils.data as D
import torchvision
import torchvision.transforms as transforms
from verifier_strategy import VerifierStrategy
from strategies.verifiers.conv_net import ConvNet


class CNNVerifierStrategy(VerifierStrategy):
    def __init__(self):
        self.__batch_size = 4
        self.__device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # normalize dtw matrices to [-1, 1] range
        self.__transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.__net = None

    def train(self, training_data: Any, classes: Tuple) -> None:
        n_epochs = 4
        learning_rate = 0.001

        # temporary override of training_data with CIFAR10 dataset to check
        # if the architecture works before running it on DTW matrices
        training_data = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                     download=True, transform=self.__transform)
        train_loader = D.DataLoader(training_data, batch_size=self.__batch_size,
                                    shuffle=True)

        self.__net = ConvNet().to(self.__device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.__net.parameters(), lr=learning_rate)

        n_total_steps = len(train_loader)
        for epoch in range(n_epochs):
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(self.__device)
                labels = labels.to(self.__device)

                outputs = self.__net(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i + 1) % 2000 == 0:
                    print(f'Epoch: {epoch + 1}/{n_epochs}, step: {i + 1}/{n_total_steps}, loss: {loss.item():.3f}')

        print('Training has finished.')

    def verify(self, samples: Any, classes: Tuple) -> float:
        samples = torchvision.datasets.CIFAR10('root=./data', train=False,
                                               download=True, transform=self.__transform)
        test_loader = D.DataLoader(samples, batch_size=self.__batch_size,
                                   shuffle=False)

        # temporary override of classes with CIFAR10 classes
        # to check if the architecture works
        classes = ('plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')

        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            n_class_correct = [0] * 10
            n_class_samples = [0] * 10
            for images, labels in test_loader:
                images = images.to(self.__device)
                labels = labels.to(self.__device)
                outputs = self.__net(images)
                _, predicted = torch.max(outputs, 1)
                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()

                for i in range(self.__batch_size):
                    label = labels[i]
                    pred = predicted[i]
                    if label == pred:
                        n_class_correct[label] += 1
                    n_class_samples[label] += 1

            acc = 100.0 * n_correct / n_samples
            print(f'Accuracy: {acc} %')

            for i in range(10):
                class_acc = 100.0 * n_class_correct[i] / n_class_samples[i]
                print(f'Accuracy: {classes[i]}: {class_acc} %')

        return acc
