"""
This example uses a very simple model and the MNIST dataset to show how the model,
the training and validation datasets, as well as the training and testing loops can
be customized in Plato.

It specifically uses Catalyst, a popular deep learning framework, to implement the
training and testing loop.
"""
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from catalyst import dl
from catalyst.data.transforms import ToTensor
from catalyst.contrib.datasets import MNIST

os.environ['config_file'] = 'configs/MNIST/fedavg_lenet5.yml'

from clients import simple
from datasources import base
from servers import fedavg
from trainers import basic


class DataSource(base.DataSource):
    """A custom datasource with custom training and validation
       datasets.
    """
    def __init__(self):
        super().__init__()

        self.trainset = MNIST('./data',
                              train=True,
                              download=True,
                              transform=ToTensor())
        self.testset = MNIST('./data',
                             train=False,
                             download=True,
                             transform=ToTensor())


class Trainer(basic.Trainer):
    """A custom trainer with custom training and testing loops. """
    def train_model(self, config, trainset, sampler, cut_layer=None):  # pylint: disable=unused-argument
        """A custom training loop. """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.02)
        train_loader = DataLoader(dataset=trainset,
                                  batch_size=config['batch_size'],
                                  sampler=sampler)

        # Training the model using Catalyst's SupervisedRunner
        runner = dl.SupervisedRunner()

        runner.train(model=self.model,
                     criterion=criterion,
                     optimizer=optimizer,
                     loaders={"train": train_loader},
                     num_epochs=1,
                     logdir="./logs",
                     verbose=True)

    def test_model(self, config, testset):  # pylint: disable=unused-argument
        """A custom testing loop. """
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=config['batch_size'], shuffle=False)

        correct = 0
        total = 0

        # Using Catalyst's SupervisedRunner and AccuracyCallback to compute accuracies
        runner = dl.SupervisedRunner()
        runner.train(model=self.model,
                     num_epochs=1,
                     loaders={"valid": test_loader},
                     logdir="./logs",
                     verbose=True,
                     callbacks=[
                         dl.AccuracyCallback(input_key="logits",
                                             target_key="targets",
                                             num_classes=10)
                     ])

        # Computing top-1 accuracy
        with torch.no_grad():
            for examples, labels in test_loader:
                examples, labels = examples.to(self.device), labels.to(
                    self.device)

                examples = examples.view(len(examples), -1)
                outputs = self.model(examples)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        return accuracy


def main():
    """A Plato federated learning training session using a custom model. """

    model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))

    datasource = DataSource()
    trainer = Trainer(model=model)

    client = simple.Client(model=model, datasource=datasource, trainer=trainer)
    server = fedavg.Server(model=model)
    server.run(client)


if __name__ == "__main__":
    main()
