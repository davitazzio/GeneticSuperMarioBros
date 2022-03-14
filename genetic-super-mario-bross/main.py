import argparse
import os
import threading

from genetic_algorithms.population import Population
from parameters import Params
from queue import Queue
from neural_network.chromosome import Chromosome


def run_population(population: Population, param: Params):
    q = Queue()
    for ch in population.get_chromosomes():
        ch.set_queue(q)
        t = threading.Thread(target=ch.run_chromosome())
        t.start()

    for i in range(0, len(population.get_chromosomes())):
        q.get()
        # print('get ', i)

    print("finished")

    population.calc_stats()
    # ecc ecc


def next_generation(population: Population) -> None:
    """

    :param population:
    :return:
    if args.debug:
        print(f'----Current Gen: {self.current_generation}, True Zero: {self._true_zero_gen}')
        fittest = self.population.fittest_individual
        print(f'Best fitness of gen: {fittest.fitness}, Max dist of gen: {fittest.farthest_x}')
        num_wins = sum(individual.did_win for individual in self.population.individuals)
        pop_size = len(self.population.individuals)
        print(f'Wins: {num_wins}/{pop_size} (~{(float(num_wins) / pop_size * 100):.2f}%)')"""

    population.save_generation()
    population.evolve()


def parse_args():
    parser = argparse.ArgumentParser(description='Super Mario Bros AI')

    # Config
    parser.add_argument('-c', '--config', dest='config', required=False, help='config file to use')
    # Load arguments
    parser.add_argument('--load-file', dest='load_file', required=False,
                        help='/path/to/population that you want to load individuals from')
    parser.add_argument('--load-inds', dest='load_inds', required=False,
                        help='[start,stop] (inclusive) or ind1,ind2,... that you wish to load from the file')
    # No display
    parser.add_argument('--no-display', dest='no_display', required=False, default=False, action='store_true',
                        help='If set, there will be no Qt graphics displayed and FPS is increased to max')
    # Debug
    parser.add_argument('--debug', dest='debug', required=False, default=False, action='store_true',
                        help='If set, certain debug messages will be printed')
    # Replay arguments
    parser.add_argument('--replay-file', dest='replay_file', required=False, default=None,
                        help='/path/to/population that you want to replay from')
    parser.add_argument('--replay-inds', dest='replay_inds', required=False, default=None,
                        help='[start,stop] (inclusive) or ind1,ind2,ind50,... or [start,] that you wish to replay from file')

    args = parser.parse_args()

    load_from_file = bool(args.load_file) and bool(args.load_inds)
    replay_from_file = bool(args.replay_file) and bool(args.replay_inds)

    # Load from file checks
    if bool(args.load_file) ^ bool(args.load_inds):
        parser.error('--load-file and --load-inds must be used together.')
    if load_from_file:
        # Convert the load_inds to be a list
        # Is it a range?
        if '[' in args.load_inds and ']' in args.load_inds:
            args.load_inds = args.load_inds.replace('[', '').replace(']', '')
            ranges = args.load_inds.split(',')
            start_idx = int(ranges[0])
            end_idx = int(ranges[1])
            args.load_inds = list(range(start_idx, end_idx + 1))
        # Otherwise it's a list of individuals to load
        else:
            args.load_inds = [int(ind) for ind in args.load_inds.split(',')]

    # Replay from file checks
    if bool(args.replay_file) ^ bool(args.replay_inds):
        parser.error('--replay-file and --replay-inds must be used together.')
    if replay_from_file:
        # Convert the replay_inds to be a list
        # is it a range?
        if '[' in args.replay_inds and ']' in args.replay_inds:
            args.replay_inds = args.replay_inds.replace('[', '').replace(']', '')
            ranges = args.replay_inds.split(',')
            has_end_idx = bool(ranges[1])
            start_idx = int(ranges[0])
            # Is there an end idx? i.e. [12,15]
            if has_end_idx:
                end_idx = int(ranges[1])
                args.replay_inds = list(range(start_idx, end_idx + 1))
            # Or is it just a start? i.e. [12,]
            else:
                end_idx = start_idx
                for fname in os.listdir(args.replay_file):
                    if fname.startswith('best_ind_gen'):
                        ind_num = int(fname[len('best_ind_gen'):])
                        if ind_num > end_idx:
                            end_idx = ind_num
                args.replay_inds = list(range(start_idx, end_idx + 1))
        # Otherwise it's a list of individuals
        else:
            args.replay_inds = [int(ind) for ind in args.replay_inds.split(',')]

    if replay_from_file and load_from_file:
        parser.error('Cannot replay and load from a file.')

    # Make sure config AND/OR [(load_file and load_inds) or (replay_file and replay_inds)]
    if not (bool(args.config) or (load_from_file or replay_from_file)):
        parser.error('Must specify -c and/or [(--load-file and --load-inds) or (--replay-file and --replay-inds)]')

    return args


if __name__ == "__main__":
    global args
    parameters = Params()

    pop = Population(parameters, 'prova7', 40, True)
    pop.load_generation(59)

    for _ in range(0, 30):
        run_population(pop, parameters)
        next_generation(pop)

    '''ch = Chromosome(parameters.network_architecture, 'ch10', True)
    ch.load_chromosome(os.path.join('prova6', 'gen230'))
    ch.run_chromosome(True)'''
