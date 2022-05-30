"""
    @Tazzioli Davide - davide.tazzioli@studio.unibo.it
"""

import numpy as np
import torch

from nes_py.wrappers import JoypadSpace
from actions import SIMPLE_MOVEMENT
from RAM_locations import EnemyType, StaticType, DynamicType
from smb_env import SuperMarioBrosEnv
from parameters import Params


def get_distance(mario, tile):
    row = mario[0] - tile[0]
    col = tile[1] - mario[1]

    return [np.clip(row / 10, -1, 1), np.clip(col / 10, -1, 1)]


class Game:
    def __init__(self, par: Params, net: torch.nn.Sequential, render=False, target=(1, 1)):

        self.life = 2
        self.old_pos = 0
        self.old_life = 2
        self.pos = 0
        self.done = None
        self.reward = None
        self.mario = None
        self.tiles = None
        self.mario_status = 'small'
        self.old_mario_status = 'small'
        self.env = SuperMarioBrosEnv(target=target)
        self.env = JoypadSpace(self.env, SIMPLE_MOVEMENT)
        self.state = self.env.reset()
        self.net = net
        self.fitness = 0
        self.score = 0
        self.render = render
        self.old_score = 0
        self.params = par

    def start_game(self):
        """
        start a new game
        :return: fitness point and information
        """
        stuck = 0
        self.state, self.reward, self.done, info = self.env.step(1)

        for step in range(5000):
            if self.done:
                break

            self.mario = info['mario_location']
            self.tiles = info['tiles']
            self.old_mario_status = self.mario_status
            self.mario_status = info['status']
            self.old_score = self.score
            self.score = info['score']
            self.old_pos = self.pos
            self.pos = info['pos_in_level'].x
            self.old_life = self.life
            self.life = info['life']

            s = self.get_step()
            self.state, self.reward, self.done, info = self.env.step(s)
            self.calculate_fitness()

            if self.old_pos == self.pos:
                stuck += 1
            else:
                stuck = 0
            if stuck == 400:
                self.done = True
                self.fitness -= 500

            if info['is_dead']:
                self.done = True

            if self.fitness < -200:
                self.fitness = -200
                self.done = True

            if self.render:
                self.env.render()
        self.state = self.env.reset()
        self.env.close()
        return self.get_fitness(), self.life, self.pos, self.mario_status, self.score

    def get_fitness(self):
        """
        return the fitnes score at the moment
        :return: fitness score
        """
        return self.fitness

    def get_ground(self):
        """
        calculate the ground array
        :return:
        """
        ground = []
        for col in range(self.params.get_interest_area()[2], self.params.get_interest_area()[3] + 1):
            result = -1
            last = False
            for row in range(self.params.get_interest_area()[0], self.params.get_interest_area()[1] + 1):
                try:
                    tile = self.tiles[(row, col)]
                except KeyError:
                    continue
                if tile is StaticType.Empty:
                    last = False
                if tile is not StaticType.Empty and not last and not isinstance(tile, EnemyType) and \
                        tile is not StaticType.PowerUp and tile is not DynamicType.Mario:
                    result = 1 - ((row - self.params.get_interest_area()[0]) / 10)
                    last = True
            ground.append(result)
        return ground

    def get_state(self):
        """
        calculate the state
        :return:
        """
        _coinblock = list(i for i in range(0, 5))  # 5
        _coin = list(i for i in range(5, 10))  # 5
        _breackable_block = list(i for i in range(10, 15))  # 5
        _flagpole_top = [15]
        _powe_up = [16]
        _power_up_block = [17]
        _piranha_plant = list(i for i in range(18, 21))
        _goomba = list(i for i in range(21, 25))  # KOOPA, SPINY_EGG
        _paratroopa = list(i for i in range(25, 29))
        _generic_enemy = list(i for i in range(29, 33))

        state = [[] for i in range(0, 33)]
        row_len = self.params.get_interest_area()[3] - self.params.get_interest_area()[2] + 1
        for p in range(0, self.params.get_input_size()):
            col = (p % row_len) + self.params.get_interest_area()[2]
            row = int(p / row_len) + self.params.get_interest_area()[0]
            try:
                tile = self.tiles[(row, col)]
            except KeyError:
                continue

            if tile is StaticType.Coin_Block1 or tile is StaticType.Hidden_Coin:
                for i in _coinblock:
                    if len(state[i]) == 0:
                        state[i] = get_distance(self.mario, (row, col))
                        break

            if tile is StaticType.Coin or \
                    tile is StaticType.Coin2:
                print('coin')
                for i in _coin:
                    if len(state[i]) == 0:
                        state[i] = get_distance(self.mario, (row, col))
                        break

            if tile is StaticType.Breakable_Block:
                for i in _breackable_block:
                    if len(state[i]) == 0:
                        state[i] = get_distance(self.mario, (row, col))
                        break

            if tile is StaticType.Flagpole_Top:
                for i in _flagpole_top:
                    if len(state[i]) == 0:
                        state[i] = get_distance(self.mario, (row, col))
                        break

            if tile is StaticType.PowerUp_Block or tile is StaticType.Hidden_life or tile is StaticType.Hidden_Powerup:
                for i in _power_up_block:
                    if len(state[i]) == 0:
                        state[i] = get_distance(self.mario, (row, col))
                        break

            if tile is StaticType.PowerUp:
                for i in _powe_up:
                    if len(state[i]) == 0:
                        state[i] = get_distance(self.mario, (row, col))
                        break

            if isinstance(tile, EnemyType):

                if tile is EnemyType.Piranha_Plant:
                    for i in _piranha_plant:
                        if len(state[i]) == 0:
                            state[i] = get_distance(self.mario, (row, col))
                            break

                elif tile is EnemyType.Goomba or \
                        tile is EnemyType.Green_Koopa2 or \
                        tile is EnemyType.Red_Koopa2 or \
                        tile is EnemyType.Spiny_Egg:
                    for i in _goomba:
                        if len(state[i]) == 0:
                            state[i] = get_distance(self.mario, (row, col))
                            break

                elif tile is EnemyType.Green_Koopa_Paratroopa or tile is EnemyType.Green_Paratroopa_Jump:
                    for i in _paratroopa:
                        if len(state[i]) == 0:
                            state[i] = get_distance(self.mario, (row, col))
                            break

                elif tile is EnemyType.Green_Koopa1 or tile is EnemyType.Red_Koopa1:
                    break

                else:
                    for i in _generic_enemy:
                        if len(state[i]) == 0:
                            state[i] = get_distance(self.mario, (row, col))
                            break
        if self.mario_status == 'tall':
            state.append([0, 1])
        elif self.mario_status == 'fireball':
            state.append([1, 1])
        else:
            state.append([0, 0])
        j = 0
        for i in range(0, len(state)):
            if len(state[i]) == 0:
                state[i] = [0, 0]
                j += 1

        return state

    def calculate_fitness(self):
        """
        Calculate the fitness score
        :return:
        """
        self.fitness += self.reward
        self.fitness += (self.score - self.old_score)
        if self.life - self.old_life > 0:
            self.fitness += 1000
            print('vita!')
        if self.mario_status == 'tall' and self.old_mario_status == 'small':
            self.fitness += 2000
            print('TALL')
        if self.mario_status == 'fireball' and self.old_mario_status == 'tall':
            self.fitness += 5000
            print('FIREBALL')
        if self.mario_status == 'small' and self.old_mario_status == 'fireball':
            self.fitness -= 1000
        if self.mario_status == 'small' and self.old_mario_status == 'tall':
            self.fitness -= 1000

    def get_step(self):
        """
        Calculate the forward step
        :return:
        """
        state = self.get_ground() + sum(self.get_state(), [])

        result = self.net.forward(torch.tensor(state, dtype=torch.float32))
        out = (result < 1.3).int().numpy()
        out_str = str(str(out[0]) + str(out[1]) + str(out[2]) + str(out[3]) + '0' + '0' + str(out[4]) + str(out[5]))
        return int(out_str, 2)


if __name__ == "__main__":
    p = Params()
    r = Game(p, None, True)
    r.start_game()
