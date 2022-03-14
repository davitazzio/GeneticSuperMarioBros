import torch

from nes_py.wrappers import JoypadSpace
from actions import SIMPLE_MOVEMENT
from RAM_locations import EnemyType, StaticType, DynamicType
from smb_env import SuperMarioBrosEnv


class Game:
    def __init__(self, net: torch.nn.Sequential, render=False):
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
            else:
                stuck = 0
            if stuck == 500:
                self.done = True
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
        state = []
        for i in range(0, 63):
            col = (i % 7) + 7
            row = int(i / 7) + 5
            done = False
            try:
                tile = self.tiles[(row, col)]
            except KeyError:
                state.append(0)
                done = True
                continue
            if not done:
                # ENEMY -1
                if isinstance(tile, EnemyType):
                    state.append(-1)
                # GROUND -0.5
                elif isinstance(tile, StaticType) and tile is not StaticType.Coin \
                        and tile is not StaticType.Coin_Block1 \
                        and tile is not StaticType.Coin_Block2:
                    if tile is StaticType.Empty or tile is StaticType.Fake:
                        state.append(0)
                    else:
                        state.append(-0.5)
                # COIN 0.5
                elif tile is StaticType.Coin \
                        or tile is StaticType.Coin_Block1 \
                        or tile is StaticType.Coin_Block2:
                    state.append(0.5)
                # POWER UP 1
                elif tile is DynamicType.PowerUp:
                    state.append(1)
                else:
                    state.append(0)

        return state

    def calculate_fitness(self):
        self.fitness += self.reward
        score_gained = self.score - self.old_score
        self.fitness += score_gained
        if self.mario_status == 'tall' and self.old_mario_status == 'small':
            self.fitness += 1000
            print('TALL')
        if self.mario_status == 'fireball' and self.old_mario_status == 'tall':
            self.fitness += 2000
            print('FIREBALL')
        if self.mario_status == 'tall' and self.old_mario_status == 'fireball':
            self.fitness -= 2000
        if self.mario_status == 'small' and self.old_mario_status == 'tall':
            self.fitness -= 1000

    def get_step(self):
        return torch.argmax(self.net.forward(torch.tensor(self.get_state()))).item()


if __name__ == "__main__":
    ''' p = utils.Utils()
    p.init_network()

    r = Game(p.network, True)
    r.start_game()'''
