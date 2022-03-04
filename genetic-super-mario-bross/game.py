from nes_py.wrappers import JoypadSpace
import gym
from actions import SIMPLE_MOVEMENT
from RAM_locations import EnemyType, StaticType, DynamicType
import numpy as np


class Round:

    def __init__(self):
        self.result = 0
        self.info = None
        self.done = None
        self.reward = None
        self.mario = None
        self.tiles = None
        self.env = gym.make('SuperMarioBros-v0')
        self.env = JoypadSpace(self.env, SIMPLE_MOVEMENT)
        self.parameters = []
        self.state = self.env.reset()

    def get_fitness(self):
        return self.result

    def get_parameters(self):
        return self.parameters

    def fintnes(self):
        pass

    def get_enemy_state(self):
        enemy = []
        for key in self.tiles.keys():
            if 5 <= key[0] <= 13 and 7 <= key[1] <= 13:
                if isinstance(self.tiles[key], EnemyType):
                    enemy.append(1)
                else:
                    enemy.append(0)
        while True:
            if not len(enemy) == 63:
                enemy.append(0)
            else:
                break
        return enemy

    def get_coin_state(self):
        coins = []
        for key in self.tiles.keys():
            if 5 <= key[0] <= 13 and 7 <= key[1] <= 13:
                if self.tiles[key] is StaticType.Coin \
                        or self.tiles[key] is StaticType.Coin_Block1 \
                        or self.tiles[key] is StaticType.Coin_Block2:
                    coins.append(1)
                else:
                    coins.append(0)
        while True:
            if not len(coins) == 63:
                coins.append(0)
            else:
                break
        return coins

    def get_environment(self):
        environment = []
        for key in self.tiles.keys():
            if 5 <= key[0] <= 13 and 7 <= key[1] <= 13:
                if self.tiles[key] is StaticType.Ground \
                        or self.tiles[key] is StaticType.Top_Pipe2 \
                        or self.tiles[key] is StaticType.Top_Pipe1 \
                        or self.tiles[key] is StaticType.Bottom_Pipe1 \
                        or self.tiles[key] is StaticType.Bottom_Pipe2 \
                        or self.tiles[key] is StaticType.Breakable_Block \
                        or self.tiles[key] is StaticType.Flagpole:
                    environment.append(1)
                else:
                    environment.append(0)
        while True:
            if not len(environment) == 63:
                environment.append(0)
            else:
                break
        return environment

    def get_power_up_state(self):
        power_up = []
        for key in self.tiles.keys():
            if 5 <= key[0] <= 13 and 7 <= key[1] <= 13:
                if self.tiles[key] is DynamicType.PowerUp:
                    power_up.append(1)
                else:
                    power_up.append(0)
        while True:
            if not len(power_up) == 63:
                power_up.append(0)
            else:
                break
        return power_up

    def get_step(self):
        return None

    def start_game(self, par, queue, render=False):
        self.parameters = par
        self.state, self.reward, self.done, self.info = self.env.step(1)
        for step in range(1000000):
            if self.done:
                break
                self.state = self.env.reset()

            self.mario = self.info['mario_location']
            self.tiles = self.info['tiles']
            s = self.get_step()
            self.state, self.reward, self.done, self.info = self.env.step(s)
            self.fintnes()

            if last_pos == self.mario[1]:
                fermo = fermo + 1
            else:
                fermo = 0
            if fermo == 200:
                self.done = True
            last_pos = self.mario[1]

            if render:
                self.env.render()

        if queue is not None:
            queue.put(self.get_fitness())
            queue.task_done()
        else:
            print('Fitness score: ', self.get_fitness())

        self.env.close()
        exit(0)


if __name__ == "__main__":
    parameters=[]
    r = Round()
    r.start_game(np.array_split(parameters, 4), None, True)
