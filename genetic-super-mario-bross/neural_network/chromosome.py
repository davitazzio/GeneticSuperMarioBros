import json
import os
from queue import Queue
from typing import List

import numpy as np
from game import Game
from .my_nn import SuperMarioNeuralNetwork
from parameters import Params

class Chromosome:
    def __init__(self, par: Params, ch_name: str, init: bool = False):
        """

        :param network_architecture: List[int]: dimension of each layer
        :param ch_name: name of the individual
        :param init: if True, initialize the individual to random values
        """
        self.network_architecture = par.get_network_architecture()
        self.input_size = self.network_architecture[0]
        self.hidden_layer_size = self.network_architecture[1]
        self.output_size = self.network_architecture[2]
        self.number_of_layers = par.get_num_layers()
        self.fitness = 0
        self.weight = [0 for i in range(0, self.number_of_layers)]
        self.bias = [0 for i in range(0, self.number_of_layers)]
        self.queue = None
        self.name = ch_name
        self.param = par

        if init:
            self.init_uniform()

    def init_uniform(self) -> None:
        """
        initialize the individuals to random values uniformly distribuited between -1 and 1
        :return: None
        """
        for layer in range(1, self.number_of_layers):
            self.weight[layer] = np.random.uniform(-1, 1, size=(
                self.network_architecture[layer], self.network_architecture[layer - 1]))
            self.bias[layer] = np.random.uniform(-1, 1, size=(self.network_architecture[layer], 1))

    def get_name(self) -> str:
        """

        :return: str: the name of the individual
        """
        return self.name

    def set_name(self, ch_name: str) -> None:
        """
        set the name of the individual
        :param ch_name: str: name of the individual
        :return: None
        """
        self.name = ch_name

    def get_input_size(self) -> int:
        """
        gets the total dimension of the parameters of the individual (the name is wrong, need to be refactror)
        :return: int: dimension
        """
        size = 0
        for layer in self.weight:
            size += layer.size()
        for layer in self.bias:
            size += layer.size()
        return size

    def get_hidden_layer_size(self) -> int:
        """

        :return: int: size of hidden layer
        """
        return self.hidden_layer_size

    def get_output_size(self):
        return self.output_size

    def get_weight(self):
        return self.weight

    def get_size(self):
        return len(self.weight) + len(self.bias)

    def get_bias(self):
        return self.bias

    def set_weight(self, weight: np.array, layer: int):
        self.weight[layer] = weight
        return True

    def set_bias(self, bias: np.array, layer: int):
        self.bias[layer] = bias
        return True

    def save_chromosome(self, population_folder: str, round_name: str = None) -> None:
        """

        :param population_folder: str: folder od the population
        :param round_name: str: name of the individual
        :return: None
        """
        # Make population folder if it doesnt exist
        if round_name is None:
            round_name = self.name
        if not os.path.exists(population_folder):
            os.makedirs(population_folder)

        ch_info = {'w1': self.weight[1].tolist(), 'w2': self.weight[2].tolist(), 'b1': self.bias[1].tolist(),
                   'b2': self.bias[2].tolist(),
                   'fitness': float(self.fitness)}

        with open(os.path.join(population_folder, round_name), 'w') as json_file:
            json.dump(ch_info, json_file)
        json_file.close()

    def load_chromosome(self, population_folder: str, individual_name: str = None) -> None:
        """

        :param population_folder: str:
        :param individual_name: str:
        :return: None
        """
        if individual_name is None:
            individual_name = self.name
        # Make sure individual exists inside population folder
        file_name = os.path.join(population_folder, individual_name)
        if not os.path.exists(file_name):
            raise Exception(f'{individual_name} not found inside {population_folder}')

        try:
            with open(file_name) as json_file:
                data = json.load(json_file)
        except:
            print('Error to load')
            data = None
        if data is not None:
            self.weight[1] = np.array(data['w1'])
            self.weight[2] = np.array(data['w2'])
            self.bias[1] = np.array(data['b1'])
            self.bias[2] = np.array(data['b2'])
            self.fitness = data['fitness']

    def get_fitness(self):
        return self.fitness

    def run_chromosome(self, render: bool = False):
        """

        :param render: bool: render the game
        :return: None
        """
        #print('chromosome ', self.name, ' started')
        nn = SuperMarioNeuralNetwork(self.input_size, self.hidden_layer_size, self.output_size)
        nn.set_params(self)
        game = Game(self.param, nn, render)
        self.fitness = game.start_game()
        self.queue.put(0)
        self.queue = None

    def get_queue(self):
        return self.queue

    def set_queue(self, queue: Queue):
        self.queue = queue
