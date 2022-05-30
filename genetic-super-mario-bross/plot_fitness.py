'''
    @Tazzioli Davide - davide.tazzioli@studio.unibo.it
'''
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
                stats[int(extension[1])] = [data['mean'], data['median'], data['std'], data['min'], data['max'], data['execution_time']]
                generations.append(int(extension[1]))

    for i in range(np.min(generations), np.max(generations)):
        if stats.get(i) is not None:
            mean.append(stats.get(i)[0])
            median.append(stats.get(i)[1])
            std.append(stats.get(i)[2])
            _min.append(stats.get(i)[3])
            _max.append(stats.get(i)[4])
            _time.append(stats.get(i)[5])

    plt.subplot(211)
    plt.plot([i for i in range(len(_max))], _max)
    plt.ylabel(f"Average Fitness by generation")
    plt.xlabel("Generation #")

    plt.subplot(212)
    plt.plot([i for i in range(len(_time))], _time)
    plt.show()


if __name__ == "__main__":
    load_stats('prova1')
