import torch
import torch.nn as nn


class SuperMarioNeuralNetwork(nn.Sequential):
    def __init__(self, input_size, hidden1_size, num_classes):
        self.input_size = input_size
        super(SuperMarioNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size, bias=True)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1_size, num_classes, bias=True)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        return out

    def get_input_size(self):
        return self.input_size

    def set_params(self, chromosome):
        '''
        Sets the parameters for the networks
        @params
            chromosome : parameters
        '''
        weights = chromosome.get_weight()
        bias = chromosome.get_bias()
        self.fc1.weight = torch.nn.Parameter(torch.Tensor(weights[1]))
        self.fc2.weight = torch.nn.Parameter(torch.Tensor(weights[2]))

        self.fc1.bias = torch.nn.Parameter(torch.Tensor(bias[1][0]))
        self.fc2.bias = torch.nn.Parameter(torch.Tensor(bias[2][0]))


