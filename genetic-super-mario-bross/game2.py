import random

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
    def __init__(self, par: Params, net: torch.nn.Sequential, render=False):
        self.done = None
        self.reward = None
        self.mario = None
        self.tiles = None
        self.mario_status = 'small'
        self.old_mario_status = 'small'
        self.env = SuperMarioBrosEnv()
        self.env = JoypadSpace(self.env, SIMPLE_MOVEMENT)
        self.state = self.env.reset()
        self.net = net
        self.fitness = 0
        self.score = 0
        self.render = render
        self.old_score = 0
        self.params = par

    def start_game(self) -> float:
        stuck = 0
        last_pos = 0
        self.state, self.reward, self.done, info = self.env.step(1)
        for step in range(100000):
            if self.done:
                break
                self.state = self.env.reset()

            self.mario = info['mario_location']
            self.tiles = info['tiles']
            self.old_mario_status = self.mario_status
            self.mario_status = info['status']
            self.old_score = self.score
            self.score = info['score']

            s = self.get_step()
            self.state, self.reward, self.done, info = self.env.step(s)
            self.calculate_fitness()

            if last_pos == self.mario[1]:
                stuck += 1
                if s == 1 or s == 6 or s==3:
                    self.fitness -= 10
            else:
                stuck = 0
            if stuck == 1000:
                self.done = True
                self.fitness -= 500
            last_pos = self.mario[1]

            if info['is_dead']:
                self.done = True

            if self.render:
                self.env.render()
        # print('Fitness score: ', self.get_fitness())
        self.env.close()
        return self.get_fitness()

    def get_fitness(self):
        return self.fitness

    def get_state(self):

        COINBLOCK = list(i for i in range(0, 6))  # 6
        COIN = list(i for i in range(6, 12))  # 6
        BREAKABLE_BLOCK = list(i for i in range(12, 18))  # 6
        FLAGPOLE_TOP = [19]
        POWER_UP = [20]
        POWER_UP_BLOCK = [21]
        SPRING = [23]
        JUMP_SPRING = [24]
        HORIZONTAL_MOVING_LIFT = [25]
        FALLING_STATIC_LIFT = list(i for i in range(26, 28))
        HORIZONTAL_LIFT = list(i for i in range(28, 30))
        VERTICAL_LIFT = list(i for i in range(30, 32))
        STATIC_LIFT = list(i for i in range(32, 34))
        FLY_CHEAP = list(i for i in range(34, 39))  # 5
        BOWSER_FLAME = list(i for i in range(39, 42))  # insieme a bullet
        PIRANHA_PLANT = list(i for i in range(42, 47))
        GOOMBA = list(i for i in range(47, 52))
        CHEEP_CHEEP = list(i for i in range(52, 57))  # pesci
        HAMMER_BROTHER = list(i for i in range(57, 60))  # ninka
        PARATROOPA = list(i for i in range(60, 65))
        KOOPA = list(i for i in range(65, 70))
        SPINY_EGG = list(i for i in range(70, 75))
        GENERIC_ENEMY = list(i for i in range(75, 80))
        HOLE = list(i for i in range(80, 85))
        PIPE = list(i for i in range(85, 90))

        state = [[] for i in range(0, 90)]
        row_len = self.params.get_interest_area()[3] - self.params.get_interest_area()[2] + 1
        for p in range(0, self.params.get_input_size()):
            col = (p % row_len) + self.params.get_interest_area()[2]
            row = int(p / row_len) + self.params.get_interest_area()[0]
            done = False
            try:
                tile = self.tiles[(row, col)]
            except KeyError:
                print('error')
                # state.append([0, 0])
                done = True
                continue
            if not done:
                if row == 13 and tile is StaticType.Empty:
                    #print('hole')
                    for i in HOLE:
                        if len(state[i]) == 0:
                            state[i] = get_distance(self.mario, (row, col))
                            break
                if tile is StaticType.Top_Pipe2 or tile is StaticType.Top_Pipe4:
                    #print('pipe ', row, col)
                    for i in PIPE:
                        if len(state[i]) == 0:
                            state[i] = get_distance(self.mario, (row, col))
                            break
                if tile is StaticType.Coin_Block1 or tile is StaticType.Hidden_Coin:
                    #print('coinblock')
                    for i in COINBLOCK:
                        if len(state[i]) == 0:
                            state[i] = get_distance(self.mario, (row, col))
                            break

                if tile is StaticType.Coin:
                    #print('coin')
                    for i in COIN:
                        if len(state[i]) == 0:
                            state[i] = get_distance(self.mario, (row, col))
                            break

                if tile is StaticType.Breakable_Block:
                    #print('breakable block')
                    for i in BREAKABLE_BLOCK:
                        if len(state[i]) == 0:
                            state[i] = get_distance(self.mario, (row, col))
                            break

                if tile is StaticType.Flagpole_Top:
                    for i in FLAGPOLE_TOP:
                        if len(state[i]) == 0:
                            state[i] = get_distance(self.mario, (row, col))
                            break

                if tile is StaticType.PowerUp_Block or tile is StaticType.Hidden_life or tile is StaticType.Hidden_Powerup:
                    #print('power-up')
                    for i in POWER_UP_BLOCK:
                        if len(state[i]) == 0:
                            state[i] = get_distance(self.mario, (row, col))
                            break

                if tile is StaticType.PowerUp:
                    #print('power-up')
                    for i in POWER_UP:
                        if len(state[i]) == 0:
                            state[i] = get_distance(self.mario, (row, col))
                            break

                if tile is DynamicType.Spring2:
                    for i in SPRING:
                        if len(state[i]) == 0:
                            state[i] = get_distance(self.mario, (row, col))
                            break

                if tile is DynamicType.Jump_Spring:
                    for i in JUMP_SPRING:
                        if len(state[i]) == 0:
                            state[i] = get_distance(self.mario, (row, col))
                            break

                if tile is DynamicType.Horizontal_Moving_Lift:
                    for i in HORIZONTAL_MOVING_LIFT:
                        if len(state[i]) == 0:
                            state[i] = get_distance(self.mario, (row, col))
                            break

                if tile is DynamicType.Falling_Static_Lift:
                    for i in FALLING_STATIC_LIFT:
                        if len(state[i]) == 0:
                            state[i] = get_distance(self.mario, (row, col))
                            break

                if tile is DynamicType.Horizontal_Lift:
                    for i in HORIZONTAL_LIFT:
                        if len(state[i]) == 0:
                            state[i] = get_distance(self.mario, (row, col))
                            break

                if tile is DynamicType.Vertical_Lift2:
                    for i in VERTICAL_LIFT:
                        if len(state[i]) == 0:
                            state[i] = get_distance(self.mario, (row, col))
                            break

                if tile is DynamicType.Static_Lift2:
                    for i in STATIC_LIFT:
                        if len(state[i]) == 0:
                            state[i] = get_distance(self.mario, (row, col))
                            break

                if isinstance(tile, EnemyType):
                    if tile is EnemyType.Fly_Cheep_Cheep:
                        for i in FLY_CHEAP:
                            if len(state[i]) == 0:
                                state[i] = get_distance(self.mario, (row, col))
                                break

                    elif tile is EnemyType.Bowser_Flame2 or tile is EnemyType.Bullet_Bill:
                        for i in BOWSER_FLAME:
                            if len(state[i]) == 0:
                                state[i] = get_distance(self.mario, (row, col))
                                break

                    elif tile is EnemyType.Piranha_Plant:
                        for i in PIRANHA_PLANT:
                            if len(state[i]) == 0:
                                state[i] = get_distance(self.mario, (row, col))
                                break

                    elif tile is EnemyType.Goomba:
                        #print('goomba')
                        for i in GOOMBA:
                            if len(state[i]) == 0:
                                state[i] = get_distance(self.mario, (row, col))
                                break

                    elif tile is EnemyType.Red_Cheep_Cheep or tile is EnemyType.Grey_Cheep_Cheep:
                        for i in CHEEP_CHEEP:
                            if len(state[i]) == 0:
                                state[i] = get_distance(self.mario, (row, col))
                                break

                    elif tile is EnemyType.Hammer_Brother:
                        for i in HAMMER_BROTHER:
                            if len(state[i]) == 0:
                                state[i] = get_distance(self.mario, (row, col))
                                break

                    elif tile is EnemyType.Green_Koopa_Paratroopa or tile is EnemyType.Green_Paratroopa_Jump:
                        for i in PARATROOPA:
                            if len(state[i]) == 0:
                                state[i] = get_distance(self.mario, (row, col))
                                break

                    elif tile is EnemyType.Green_Koopa2 or tile is EnemyType.Red_Koopa2:
                        for i in KOOPA:
                            if len(state[i]) == 0:
                                state[i] = get_distance(self.mario, (row, col))
                                break

                    elif tile is EnemyType.Green_Koopa1 or tile is EnemyType.Red_Koopa1:
                        break

                    elif tile is EnemyType.Spiny_Egg or tile is EnemyType:
                        for i in SPINY_EGG:
                            if len(state[i]) == 0:
                                state[i] = get_distance(self.mario, (row, col))
                                break
                    else:
                        for i in GENERIC_ENEMY:
                            print('generic enemy')
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
        #print(j)
        return state

    def calculate_fitness(self):
        self.fitness += self.reward
        score_gained = self.score - self.old_score
        self.fitness += score_gained
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
        return torch.argmax(self.net.forward(torch.tensor(sum(self.get_state(), []), dtype=torch.float32))).item()


if __name__ == "__main__":
    p = Params()
    r = Game(p, None, True)
    r.start_game()
