import random


class Params:
    def __init__(self):

        self.environment_area = ([8, 13, 9, 13])
        self.enemy_area = ([7, 12, 9, 12])
        self.coin_area = ([7, 12, 8, 12])

        self.environment_area_size = (self.environment_area[1] - self.environment_area[0] + 1) * \
                                     (self.environment_area[3] - self.environment_area[2] + 1)
        self.enemy_area_size = (self.enemy_area[1] - self.enemy_area[0] + 1) * \
                               (self.enemy_area[3] - self.enemy_area[2] + 1)
        self.coin_area_size = (self.coin_area[1] - self.coin_area[0] + 1) * \
                              (self.coin_area[3] - self.coin_area[2] + 1)
        #self.interest_area = ([6, 12, 8, 13])
        self.hidden_layer_architecture = 9
        self.num_classes = 7
        self.input_size = self.environment_area_size + self.enemy_area_size + self.coin_area_size
        #self.input_size = (self.interest_area[1] - self.interest_area[0] + 1) * (
                #self.interest_area[3] - self.interest_area[2] + 1)

        self.network_architecture = [self.input_size]  # Input Nodes
        self.network_architecture.append(self.hidden_layer_architecture)  # number of hidden layer nodes
        self.network_architecture.append(self.num_classes)  # 7 Outputs ['v', '>', '<', '^', 'a', 'b', 'none']
        self.selection_type = 'tournament'
        self.tournament_size = 5
        self.num_parents = 8
        self.number_layers = 3
        self.sbx_eta = 100
        self.mutation_rate = 0.1 # Value must be between [0.00, 1.00)
        self.gaussian_mutation_scale = 0.2 # The amount to multiple by the guassian(0, 1) value by

    def get_selection_type(self):
        return self.selection_type

    def get_tournament_size(self):
        return self.tournament_size

    def get_num_parents(self):
        return self.num_parents

    def get_num_layers(self):
        return self.number_layers

    def get_sbx_eta(self):
        #return random.randint(30, 100)
        return self.sbx_eta

    def get_mutation_rate(self):
        return self.mutation_rate

    def get_gaussian_mutation_scale(self):
        return self.gaussian_mutation_scale

    def get_interest_area(self):
        return self.interest_area

    def get_hidden_layer_architecture(self):
        return self.hidden_layer_architecture

    def get_num_classes(self):
        return self.num_classes

    def get_input_size(self):
        return self.input_size

    def get_network_architecture(self):
        return self.network_architecture
