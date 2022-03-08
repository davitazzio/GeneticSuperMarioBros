from typing import List

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch import device
from torch.autograd import Variable
from torch.nn.init import uniform_


class IrisNet(nn.Module):
    def _init_(self, input_size, hidden1_size, num_classes):
        super(IrisNet, self).init_()
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.relul = nn.ReLU()

        self.fc2 = nn.Linear(hidden1_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relul(out)
        out = self.fc2(out)
        return out


def set_params(net: torch.nn.Sequential, params: Parameters) -> torch.nn.Sequential:
    '''Sets the parameters for an nn.Sequential
    @params
        network (torch.nn.Sequential): A network to change the parameters of
        params (Parameters) : Parameters to place into the model
    @returns
        torch.nn.Sequential: A model the the provided parameters'''

    i = 0
    for layerid, layer in enumerate(net):
        if hasattr(layer, 'weight') and layer.weight is not None:
            net[layerid].weight = params[i]
            i += 1
        if hasattr(layer, 'bias') and layer.bias is not None:
            net[layerid].bias = params[i]
            i += 1
    return net


def crossover(parent1: Parameters, pop: List[Parameters]) -> Parameters:
    '''Crossover two individuals and produce a child.
    This is done by randomly splitting the weights and biases at each layer for the parents and then
    combining them to produce a child
    @params
    @returns
    parent1 (Parameters): A parent that may potentially be crossed over
    pop (List[Parameters]): The population of solutions
    Parameters: A child with attributes of both parents or the original parent1'''
    if np.random.rand() < CROSS_RATE:
        i = np.random.randint(0, POP_SIZE, size=1)[0]
        parent2 = pop[i]
        child = []
        split = np.random.rand()
        for p1l, p21 in zip(parent1, parent2):
            splitpoint = int(len(p1l) * split)
            new_param = nn.parameter.Parameter(torch.cat([p1l[:splitpoint], p2l[splitpoint:]]))
            child.append(new_param)
        return child
    else:
        return parent1


def select(pop: List[Parameters], fitnesses: np.ndarray) -> List[Parameters]:
    '''Select a new population
    @params
    @returns
    pop (List[Parametersl): The entire population of parameters
    fitnesses (np.ndarray): the fitnesses for each entity in the population
    List[Parameters]: A new population made of fitter individuals'''
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True, p=fitnesses / fitnesses.sum())
    return [pop[i] for i in idx]


def gen_mutate(shape: torch.Size) -> torch.tensor:
    '''Generate a tensor to use for random mutation of a parameter
    @params
    @returns
    shape (torch.Size): The shape of the tensor to be created
    torch.tensor: a random tensor'''
    return torch.randn(shape).to(device) * MUTATION_FACTOR


def mutate(child: Parameters) -> Parameters:
    '''Mutate a child
    @params
         child (Parameters): The original parameters
    @returns
        Parameters: The mutated child'''
    if np.random.rand() < MUTATION_RATE:
        for i in range(len(child)):
            for j in range(len(child[i])):
                child[i][j] += gen_mutate(child[i][j].shape)
    return child


# build a population
pop = []
for i in range(POP_SIZE) :
    entity= []
    for shape in shapes:
        try:
            rand_tensor = nn.init.kaiming_uniform_(torch.empty(shape)).to (device)
        except ValueError:
            rand_tensor = nn.init.uniform_(torch.empty(shape), -0.2, 0.2). to (device)

        entity.append((torch.nn.parameter.Parameter(rand_tensor)))
    pop.append(entity)