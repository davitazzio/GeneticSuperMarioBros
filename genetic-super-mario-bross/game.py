import torch
from nes_py.wrappers import JoypadSpace
from actions import SIMPLE_MOVEMENT
from RAM_locations import EnemyType, StaticType, DynamicType
from smb_env import SuperMarioBrosEnv
from parameters import Params


class Game:
    def __init__(self, parameters: Params, net: torch.nn.Sequential, render=False, target=(1, 1)):
        self.env = SuperMarioBrosEnv(target=target)
        self.env = JoypadSpace(self.env, SIMPLE_MOVEMENT)
        self.state = self.env.reset()
        self.params = parameters
        self.net = net
        self.fitness = 0
        self.score = 0
        self.render = render
        self.old_score = 0
        self.best_status = 'small'
        self.done = None
        self.reward = None
        self.mario = None
        self.tiles = None
        self.mario_status = 'small'
        self.old_mario_status = 'small'

    def start_game(self) -> float:
        stuck = 0
        last_pos = 0
        self.state, self.reward, self.done, info = self.env.step(1)
        for step in range(50000):
            if self.done:
                break

            self.mario = info['mario_location']
            self.tiles = info['tiles']
            self.old_mario_status = self.mario_status
            self.mario_status = info['status']
            self.old_score = self.score
            self.score = info['score']

            self.state, self.reward, self.done, info = self.env.step(self.get_step())
            self.calculate_fitness()

            if last_pos == self.mario[1]:
                stuck += 1
            else:
                stuck = 0
            if stuck == 500:
                self.done = True
                self.fitness -= 500
            last_pos = self.mario[1]
            if info['is_dead']:
                self.done = True
                self.fitness -= 500
            if self.render:
                self.env.render()

        self.state = self.env.reset()
        self.env.close()
        return self.get_statistics()

    def get_fitness(self):
        return self.fitness

    def get_environment(self):
        environment = []
        row_len = self.params.environment_area[3] - self.params.environment_area[2] + 1
        for p in range(0, self.params.environment_area_size):
            col = (p % row_len) + self.params.environment_area[2]
            row = int(p / row_len) + self.params.environment_area[0]
            try:
                tile = self.tiles[(row, col)]
            except KeyError:
                continue
            if isinstance(tile, StaticType):
                if tile is StaticType.Empty:
                    environment.append(0)
                else:
                    environment.append(1)
            else:
                environment.append(1)
        return environment

    def get_enemy(self):
        enemy = []
        row_len = self.params.enemy_area[3] - self.params.enemy_area[2] + 1
        for p in range(0, self.params.enemy_area_size):
            col = (p % row_len) + self.params.enemy_area[2]
            row = int(p / row_len) + self.params.enemy_area[0]
            try:
                tile = self.tiles[(row, col)]
            except KeyError:
                continue
            if isinstance(tile, EnemyType):
                if tile is EnemyType.Goomba or \
                        tile is EnemyType.Green_Koopa2 or \
                        tile is EnemyType.Red_Koopa2 or \
                        tile is EnemyType.Spiny_Egg or \
                        tile is EnemyType.Green_Koopa_Paratroopa or \
                        tile is EnemyType.Green_Paratroopa_Jump:
                    enemy.append(0.5)
                else:
                    enemy.append(1)
            else:
                enemy.append(0)
        return enemy

    def get_coins(self):
        coins = []
        row_len = self.params.coin_area[3] - self.params.coin_area[2] + 1
        for p in range(0, self.params.coin_area_size):
            col = (p % row_len) + self.params.coin_area[2]
            row = int(p / row_len) + self.params.coin_area[0]
            try:
                tile = self.tiles[(row, col)]
            except KeyError:
                continue
            if isinstance(tile, StaticType):
                if tile is StaticType.Coin_Block1 or \
                        tile is StaticType.Hidden_Coin or \
                        tile is StaticType.Coin or \
                        tile is StaticType.PowerUp_Block or \
                        tile is StaticType.Hidden_life or \
                        tile is StaticType.Hidden_Powerup or \
                        tile is StaticType.PowerUp:
                    coins.append(1)
                else:
                    coins.append(0)
            else:
                coins.append(0)
        return coins

    def get_state(self):
        return self.get_environment() + self.get_enemy() + self.get_coins()

    def calculate_fitness(self):
        self.fitness += self.reward
        self.fitness += self.score - self.old_score
        if self.mario_status == 'tall' and self.old_mario_status == 'small':
            self.fitness += 1500
            self.best_status = 'tall'
            print('TALL')
        if self.mario_status == 'fireball' and self.old_mario_status == 'tall':
            self.fitness += 3000
            print('FIREBALL')
            self.best_status = 'fireball'
        if self.mario_status == 'small' and self.old_mario_status == 'fireball':
            self.fitness -= 500
        if self.mario_status == 'small' and self.old_mario_status == 'tall':
            self.fitness -= 500

    def get_statistics(self):
        return (
            self.get_fitness(),
            self.mario[1],
            self.best_status,
            self.score
        )

    def get_step(self):
        net_output = self.net.forward(torch.tensor(self.get_state(), dtype=torch.float32))
        out = (net_output < 1).int().numpy()
        result_str = str(str(out[0]) + str(out[1]) + str(out[2]) + str(out[3]) + '0' + '0' + str(out[4]) + str(
            out[5]))

        ''''right':  0b10000000,
        'left':   0b01000000,
        'down':   0b00100000,
        'up':     0b00010000,
        'start':  0b00001000,
        'select': 0b00000100,
        'B':      0b00000010,
        'A':      0b00000001,
        'NOOP':   0b00000000,'''

        return int(result_str, 2)


if __name__ == "__main__":
    p = Params()
    r = Game(p, None, True)
    r.start_game()
