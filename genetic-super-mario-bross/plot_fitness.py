import json
import os

import numpy as np
from matplotlib import pyplot as plt


def load_stats(pop_name: str):
    mean = []
    median = []
    std = []
    _min = []
    _max = []
    stats = {}
    _time = []
    generations = []

    for files in os.listdir(pop_name):
        extension = files.rsplit('gen', 1)
        if extension[0] == '':
            file_name = os.path.join(os.path.join(pop_name, files), 'stat')

            try:
                with open(file_name) as json_file:
                    data = json.load(json_file)
            except:
                print('Error to load')
                data = None
            if data is not None:
                stats[int(extension[1])] = [data['mean'], data['median'], data['std'], data['min'], data['max']]
                generations.append(int(extension[1]))

    for i in range(np.min(generations), np.max(generations)):
        if stats.get(i) is not None:
            mean.append(stats.get(i)[0])
            median.append(stats.get(i)[1])
            std.append(stats.get(i)[2])
            _min.append(stats.get(i)[3])
            _max.append(stats.get(i)[4])

    plt.subplot(211)
    plt.plot([i for i in range(len(mean))], mean)
    plt.plot([i for i in range(len(_max))], _max)
    plt.ylabel(f"Average Fitness by generation")
    plt.xlabel("Generation #")

    plt.subplot(212)
    plt.plot([i for i in range(len(_time))], _time)
    #plt.plot([i for i in range(len(std))], std)

    plt.show()


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
    print({k: v for k, v in sorted(fitness.items(), key=lambda item: item[1], reverse=True)})
    return 0  # np.amax(fitness)


load_stats('prova7')
#find_best_chromosome('prova7', 129)
'''extension = fname.rsplit('.npy', 1)
        if len(extension) == 2:
            param = extension[0]
            chromosome[param] = np.load(os.path.join(population_folder, individual_name, fname))


def load_chromosome(self, population_folder: str, individual_name: str = None):

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

plt.subplot(211)
    plt.plot([i for i in range(len(lucro_nets_media))], lucro_nets_media)
    plt.plot([i for i in range(len(best_lucro_nets))], best_lucro_nets)
    plt.ylabel(f"Average Fitness by generation")
    plt.xlabel("Generation #")

    plt.subplot(212)
    plt.plot([i for i in range(len(lucro_nets))], lucro_nets)
    # plt.plot([i for i in range(len(lucro_time))], lucro_time)

    plt.show()'''
