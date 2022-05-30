'''
    @Tazzioli Davide - davide.tazzioli@studio.unibo.it
'''

import json
import os

from parameters import Params
from neural_network.chromosome import Chromosome


def find_best_chromosome(population_name: str, generation: int):
    """
    find the best chromosome in a generation.
    :param population_name: str: name of the population
    :param generation: int:
    :return: None
    """
    for files in os.listdir(population_name):
        extension = files.rsplit('gen', 1)
        if extension[0] == '' and int(extension[1]) == generation:
            f = files
            break
    fitness = {}
    for files in os.listdir(os.path.join(pop_name, f)):
        if files != 'stat':
            try:
                with open(os.path.join(os.path.join(pop_name, f), files)) as json_file:
                    data = json.load(json_file)
            except:
                print('Error to load')
                data = None
            if data is not None:
                fitness[files] = data['fitness']
    sorted_fitness = {k: v for k, v in sorted(fitness.items(), key=lambda item: item[1], reverse=True)}
    print(sorted_fitness)

    return list(sorted_fitness.keys())[0]


if __name__ == "__main__":
    parameters = Params()
    generation = 1004
    pop_name = 'prova5'
    target = (2, 4)

    ch = Chromosome(parameters, str(find_best_chromosome(pop_name, generation)), True)

    ch.load_chromosome(os.path.join(pop_name, 'gen' + str(generation)))
    ch.run_chromosome(True, target=target)
