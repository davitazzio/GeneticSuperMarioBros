import numpy as np
import random
from typing import List

from neural_network.chromosome import Chromosome


def elitism_selection(population, num_individuals: int) -> List[Chromosome]:
    # Is this efficient? No. What would be better? Max heap. Will I change it? Probably not this time.
    individuals = sorted(population.get_chromosomes(), key=lambda chromosome: chromosome.get_fitness(), reverse=True)
    return individuals[:num_individuals]


def roulette_wheel_selection(population, num_individuals: int) -> List[Chromosome]:
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
    selection = []
    for _ in range(num_individuals):
        tournament = np.random.choice(population.get_chromosomes(), tournament_size)
        best_from_tournament = max(tournament, key=lambda chromosome: chromosome.get_fitness())
        selection.append(best_from_tournament)

    return selection
