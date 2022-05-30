'''
    @Tazzioli Davide - davide.tazzioli@studio.unibo.it
    @Chrispresso - https://github.com/Chrispresso/SuperMarioBros-AI
'''

import json
import os
import random
import threading
import time
from queue import Queue

import numpy as np
from typing import List, Tuple
from neural_network.chromosome import Chromosome
from parameters import Params
from genetic_algorithms.selection import elitism_selection, tournament_selection, roulette_wheel_selection
from genetic_algorithms.mutation import gaussian_mutation
from genetic_algorithms.crossover import simulated_binary_crossover as SBX


class Population(object):
    def __init__(self, params: Params, name: str, num_individuals: int = 10, init: bool = False,
                 chromosomes: List[Chromosome] = None):
        """
        Initialize a population
        :param params: Params: parameter of the algorithm
        :param name: str: name of the population
        :param num_individuals: int: number of individuals in the population
        :param init: bool: initialize randomly the population
        :param chromosomes: List[Chromosome]: initialize the population ith given individuals
        """
        self.best_pos = []
        self.scores = []
        self.tall = 0
        self.fireball = 0
        self.num_individuals = num_individuals
        self.chromosomes = chromosomes
        self.parameters = params
        self.generation = None
        self.name = name
        if init:
            self.init_uniform()

    def run_population(self):
        start_time = time.time()
        q = Queue()
        for ch in self.chromosomes:
            ch.set_queue(q)
            t = threading.Thread(target=ch.run_chromosome())
            t.start()

        for i in range(0, len(self.chromosomes)):
            q.get()
            # print('get ', i)

        print("Finished...")
        time_of_execution = time.time() - start_time
        self.save_generation(time_of_execution)

    def get_num_individuals(self) -> int:
        """
        gets the number of individuals in the population
        :return: int: number of individuals
        """
        return self.num_individuals

    def _rename_chromosomes(self):
        i = 0
        for ch in self.chromosomes:
            ch.set_name('ch' + str(i))
            i += 1

    def get_chromosomes(self) -> List[Chromosome]:
        """
        gets the currents individuals of the population
        :return: List of Chromosome
        """
        return self.chromosomes

    def set_chromosomes(self, chromosome: List[Chromosome]) -> None:
        """
        Set the individuals of the population
        :param chromosome: List of Chromosome
        :return: None
        """
        self.chromosomes = chromosome

    def init_uniform(self) -> None:
        """
        Initialize the population of chromosomes
        with random values uniformly sitrubuted
        between -1 and 1
        :return None
        """
        self.generation = 0
        self.chromosomes = []
        for i in range(0, self.num_individuals):
            c = Chromosome(self.parameters, 'ch' + str(i), True)
            self.chromosomes.append(c)

    def calc_stats(self):
        """
        Calculate statistics of the chromosome: mean, median, std, min and max
        :return: dict{mean, median, std, min and max}
        """
        fitness = []
        for ch in self.chromosomes:
            fitness.append(ch.get_fitness())
        mean = np.mean(fitness)
        median = np.median(fitness)
        std = np.std(fitness)
        _min = float(min(fitness))
        _max = float(max(fitness))

        return {'mean': mean, 'median': median, 'std': std, 'min': _min, 'max': _max}

    def generation_statistics(self):
        self.best_pos = []
        self.scores = []
        self.tall = 0
        self.fireball = 0
        for ch in self.chromosomes:
            self.best_pos.append(ch.get_best_pos())
            self.scores.append(ch.get_score())
            if ch.get_best_status() == 'fireball':
                self.tall += 1
                self.fireball += 1
            elif ch.get_best_status() == 'tall':
                self.tall += 1

    def reward(self):
        best_position_reached = 0
        ch_rewarded = None
        if len(self.best_pos) > 0:
            for ch in self.chromosomes:
                if ch.get_best_pos() > best_position_reached:
                    best_position_reached = ch.get_best_pos()
                    ch_rewarded = ch
            if best_position_reached > max(self.best_pos):
                ch_rewarded.set_reward(10)
                print('position increased')
            if self.tall < 4:
                for ch in self.chromosomes:
                    if ch.get_best_status == 'tall':
                        ch.set_reward(10)
                        break
            if self.fireball < 4:
                for ch in self.chromosomes:
                    if ch.get_best_status == 'fireball':
                        ch.set_reward(13)
                        break

    def save_generation(self, time_of_execution: float = None) -> None:
        """
        Save the actual chromosomes of the population at the current generation, with their stats
        :return: None
        """
        if not os.path.exists(self.name):
            os.makedirs(self.name)
        population_folder = os.path.join(self.name, 'gen' + str(self.generation))
        if not os.path.exists(population_folder):
            os.makedirs(population_folder)
        for ch in self.chromosomes:
            ch.save_chromosome(population_folder)

        if not os.path.exists(population_folder):
            os.makedirs(population_folder)

        stats = self.calc_stats()
        stats['execution_time'] = time_of_execution
        print(stats)

        with open(os.path.join(population_folder, 'stat'), 'w') as json_file:
            json.dump(stats, json_file)
        json_file.close()

    def load_generation(self, generation: int, evolve: bool = True) -> None:
        """
        Load an old version of the population with the same name
        :param evolve: bool :automatically evolves the loaded population
        :param generation: int
        :return: none
        """
        population_folder = os.path.join(self.name, 'gen' + str(generation))
        for ch in self.chromosomes:
            try:
                ch.load_chromosome(population_folder)
            except Exception:
                continue

        self.generation = generation
        if evolve:
            self.evolve()

        # np.save(os.path.join(population_folder, 'stats'), self.calc_stats())

    def load_stats(self, path_to_stats: str):
        # TODO: not implemented
        pass

    def evolve(self) -> None:
        """
        Evolve the population:
            --select the best individuals
            --crossover between random best individual
            --mutate the result
            --the worst individual are replaced with the heirs
        :return: None
        """
        self.reward()
        reward = []
        for ch in self.chromosomes:
            if ch.reward > 0:
                reward.append(ch)
        self.generation_statistics()
        self.chromosomes = elitism_selection(self, self.parameters.get_num_parents())
        self.chromosomes.append(Chromosome(self.parameters, 'new', True))
        if len(reward) > 0:
            for ch in reward:
                if ch not in self.chromosomes:
                    self.chromosomes.append(ch)
        random.shuffle(self.chromosomes)

        while len(self.chromosomes) < self.num_individuals:
            if self.parameters.get_selection_type() == 'tournament':
                p1, p2 = tournament_selection(self, 2, self.parameters.get_tournament_size())
            elif self.parameters.get_selection_type() == 'roulette':
                p1, p2 = roulette_wheel_selection(self, 2)
            else:
                raise Exception('crossover_selection is not supported')
            L = self.parameters.get_num_layers()
            weights1 = p1.get_weight()
            bias1 = p1.get_bias()

            weights2 = p2.get_weight()
            bias2 = p2.get_bias()

            # Each W_l and b_l are treated as their own chromosome.
            c1 = Chromosome(self.parameters, 'mutated', False)
            c2 = Chromosome(self.parameters, 'mutated', False)
            for l in range(1, L):
                p1_W_l = weights1[l]
                p2_W_l = weights2[l]
                p1_b_l = bias1[l]
                p2_b_l = bias2[l]

                # Crossover
                # @NOTE: same type of crossover on the weights and the bias.
                c1_W_l, c2_W_l, c1_b_l, c2_b_l = _crossover(self.parameters, p1_W_l, p2_W_l, p1_b_l, p2_b_l)

                # Mutation
                # @NOTE: same type of mutation on the weights and the bias.
                _mutation(self.parameters, c1_W_l, c2_W_l, c1_b_l, c2_b_l)

                c1.set_weight(np.clip(c1_W_l, -1, 1), l)
                c1.set_bias(np.clip(c1_b_l, -1, 1), l)
                c2.set_weight(np.clip(c2_W_l, -1, 1), l)
                c2.set_bias(np.clip(c2_b_l, -1, 1), l)
            self.chromosomes.append(c1)
            self.chromosomes.append(c2)

        self.generation += 1
        random.shuffle(self.chromosomes)
        self._rename_chromosomes()
        for ch in self.chromosomes:
            ch.decrease_reward()
        print("Population evolved to generation ", self.generation)


def _crossover(par: Params, parent1_weights: np.ndarray, parent2_weights: np.ndarray,
               parent1_bias: np.ndarray, parent2_bias: np.ndarray) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    eta = par.get_sbx_eta()

    # SBX weights and bias
    child1_weights, child2_weights = SBX(parent1_weights, parent2_weights, eta)
    child1_bias, child2_bias = SBX(parent1_bias, parent2_bias, eta)

    return child1_weights, child2_weights, child1_bias, child2_bias


def _mutation(par: Params, child1_weights: np.ndarray, child2_weights: np.ndarray,
              child1_bias: np.ndarray, child2_bias: np.ndarray) -> None:
    mutation_rate = par.get_mutation_rate()
    scale = par.get_gaussian_mutation_scale()

    #    if par.get_mutation_rate_type() == 'dynamic':
    #        mutation_rate = mutation_rate / math.sqrt(self.current_generation + 1)
    # Mutate weights
    gaussian_mutation(child1_weights, mutation_rate, scale=scale)
    gaussian_mutation(child2_weights, mutation_rate, scale=scale)

    # Mutate bias
    gaussian_mutation(child1_bias, mutation_rate, scale=scale)
    gaussian_mutation(child2_bias, mutation_rate, scale=scale)
