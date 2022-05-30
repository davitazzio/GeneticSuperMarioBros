import argparse
import os
import threading

from genetic_algorithms.population import Population
from parameters import Params
from queue import Queue
import time
from neural_network.chromosome import Chromosome


def next_generation(population: Population) -> None:
    population.evolve()


if __name__ == "__main__":
    global args
    parameters = Params()

    pop = Population(parameters, 'prova5', 25, True)
    #pop.load_generation(848)

    for _ in range(0, 1000):
        pop.run_population(pop, parameters)
        next_generation(pop)

    '''ch = Chromosome(parameters, 'ch9', True)
    ch.load_chromosome(os.path.join('prova7', 'gen130'))
    ch.run_chromosome(True)'''
