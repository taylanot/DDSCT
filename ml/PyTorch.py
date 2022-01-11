from ..src.ml import MACHINELEARNING
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split, Subset, SubsetRandomSampler, SequentialSampler 
import tqdm
import numpy as np

class SimpleTraining(MACHINELEARNING):
    def __init__(self, dataset, config):
        super().__init__(self, config)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = dataset
        self.config = config
        self.split = self.config.get('split')

        self.network = config['additional']['network'](config['additional']\
                ['network_parameters'])

        self.network.to(self.device)
        self.optimizer = config['additional']['optimizer']\
                (self.network.parameters(),\
                config['additional']['optimizer_parameters']['lr'])

        self.loss = config['additional']['loss']()
        self.epoch = config['additional']['epoch']
        self.batch_size = 10000
    
    def adjust_dataset(self):

        """ Method: Adjust your data for desired application """

        self.input = torch.Tensor(self.dataset['Running_Variables'].values)
        self.output = torch.Tensor(self.dataset['Results'].values)
        self.size = self.input.shape[0]
        self.torch_dataset = TensorDataset(self.input,self.output)

        if self.split == None:
            self.split = 0.2
        test_size = int(self.split * self.size)
        train_size = self.size - test_size

        train_indices, test_indices = random_split\
                (self.torch_dataset,[train_size, test_size])

        self.train_sampler = SubsetRandomSampler(train_indices.indices)
        self.test_sampler = SubsetRandomSampler(test_indices.indices)

        self.train_loader = DataLoader(self.torch_dataset,\
                batch_size=self.batch_size, sampler=self.train_sampler)
        self.test_loader = DataLoader(self.torch_dataset,\
                batch_size=self.batch_size, sampler=self.test_sampler)

    def train_step(self, train_loader):

        """ Method: Training step for one epoch """

        sum_loss = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)# Decompose train_loader

            self.optimizer.zero_grad()               # Clear gradient history

            outputs = self.network(inputs)               # Feed-forward
            loss = self.loss(outputs,labels)    # Loss
            loss.backward()                     # Backpropagate
            self.optimizer.step()                    # Step towards optimum
            sum_loss += loss                    # Add the losses 

        avg_loss = sum_loss / len(train_loader) # Average the loss
        
        return avg_loss

    def test_step(self, test_loader):
        
        """ Testing step for one epoch """

        sum_loss = 0

        with torch.no_grad():                       # Cancel gradient calculation

            for inputs, labels in test_loader:  
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.network(inputs)               # Feed-forward
                loss = self.loss(outputs,labels)    # Loss
                sum_loss += loss                    # Add the losses

            avg_loss = sum_loss / len(test_loader)  # Average the loss

            return avg_loss

    def train(self):

        """ Mehtod: Tranining Loop """

        train_loss = np.zeros((self.epoch,1))
        test_loss = np.zeros((self.epoch,1))

        for epoch in tqdm.tqdm(range(self.epoch)):
            train_loss[epoch,0] = (self.train_step(self.train_loader).item())
            test_loss[epoch,0] = (self.test_step(self.test_loader).item())
        return np.hstack((train_loss, test_loss))

    def save_model(self, name):
        torch.save(self.network, name)
