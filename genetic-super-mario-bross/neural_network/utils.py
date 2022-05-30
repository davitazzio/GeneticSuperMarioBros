
import numpy as np
from typing import Tuple, Optional, Union, Set, Dict, Any, List
from . import chromosome as ch
from . import my_nn


class Utils:
    def __init__(self):
        self.chromosome = None
        self.network = None
        self.network_architecture = None
        self.inputs_as_array = None
        self.interest_area = None
        self.render = None

    def init_network(self,
                     chromosome: Optional[ch.Chromosome] = None,
                     hidden_layer_architecture: int = 9,
                     interest_area: List[int] = ([5, 13, 7, 13]),
                     ):
        self.interest_area = interest_area

        # STRUTTURA DELLA RETE
        num_inputs = (self.interest_area[1] - self.interest_area[0] + 1) * (
                self.interest_area[3] - self.interest_area[2] + 1)

        self.inputs_as_array = np.zeros((num_inputs, 1))
        self.network_architecture = [num_inputs]  # Input Nodes
        self.network_architecture.append(hidden_layer_architecture)  # number of hidden layer nodes
        self.network_architecture.append(7)  # 7 Outputs ['v', '>', '<', '^', 'a', 'b', 'none']

        self.network = my_nn.SuperMarioNeuralNetwork(num_inputs, hidden_layer_architecture, 7)

        # If chromosome is set, take it
        if chromosome:
            self.chromosome = chromosome
        else:
            self.chromosome = ch.Chromosome(self.network_architecture, True)

    def save_network(self, population_folder: str, round_name: str):
        self.chromosome.save_chromosome(population_folder, round_name)

    def load_network(self, population_folder: str, round_name: str):
        self.chromosome.load_chromosome(population_folder, round_name)

    def get_network(self):
        return self.network


