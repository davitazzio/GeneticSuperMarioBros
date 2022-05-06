import json
import os

from parameters import Params
from neural_network.chromosome import Chromosome


def find_best_chromosome(pop_name: str, generation: int):
    for files in os.listdir(pop_name):
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


parameters = Params()
generation = 1339
pop_name = 'prova1'
print(find_best_chromosome(pop_name, generation))
ch = Chromosome(parameters, str(find_best_chromosome(pop_name, generation)), True)
#ch = Chromosome(parameters, 'ch18', True)
ch.load_chromosome(os.path.join(pop_name, 'gen'+str(generation)))
#ch.load_chromosome(os.path.join(pop_name, 'gen26'))
ch.run_chromosome(True)


