"""

    @Chrispresso - https://github.com/Chrispresso/SuperMarioBros-AI
    @Tazzioli Davide - davide.tazzioli@studio.unibo.it
"""
import numpy as np
import random
from typing import List

from neural_network.chromosome import Chromosome


def elitism_selection(population, num_individuals: int) -> List[Chromosome]:
    """

    :param population: Population:
    :param num_individuals: int: number of elite selected individuals
    :return: List[Individual]
    """
    individuals = sorted(population.get_chromosomes(), key=lambda chromosome: chromosome.get_fitness(), reverse=True)
    return individuals[:num_individuals]


def roulette_wheel_selection(population, num_individuals: int) -> List[Chromosome]:
    """

    :param population: Population:
    :param num_individuals: int: number individuals selected
    :return: List[Individual]
    """
    selection = []
    wheel = sum(chromosome.get_fitness() for chromosome in population.get_chromosomes())
    for _ in range(num_individuals):
        pick = random.uniform(0, wheel)
        current = 0
        for chromosome in population.get_chromosomes():
            current += chromosome.fitness
            if current > pick:
                selection.append(chromosome)
                break

    return selection


def tournament_selection(population, num_individuals: int, tournament_size: int) -> List[Chromosome]:
    """

    :param population: Population:
    :param num_individuals: int: number of elite selected individuals
    :param tournament_size: int: size of the sub-group of individual
    :return: List[Individual]
    """
    selection = []
    for _ in range(num_individuals):
        tournament = np.random.choice(population.get_chromosomes(), tournament_size)
        best_from_tournament = max(tournament, key=lambda chromosome: chromosome.get_fitness())
        selection.append(best_from_tournament)

    return selection
