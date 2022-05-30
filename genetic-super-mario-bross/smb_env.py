"""
    An OpenAI Gym environment for Super Mario Bros.
    @Christian Kauten - https://github.com/Kautenja/gym-super-mario-bros
    @Chrispresso - https://github.com/Chrispresso/SuperMarioBros-AI
    @Tazzioli Davide - davide.tazzioli@studio.unibo.it

"""
from collections import defaultdict, namedtuple
from enum import Enum
from RAM_locations import StaticType, DynamicType, EnemyType, RAMLocations
from nes_py import NESEnv

import numpy as np
from _roms.decode_target import decode_target
from _roms.rom_path import rom_path

# create a dictionary mapping value of status register to string names
_STATUS_MAP = defaultdict(lambda: 'fireball', {0: 'small', 1: 'tall'})

# a set of state values indicating that Mario is "busy"
_BUSY_STATES = [0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x07]

# RAM addresses for enemy types on the screen
_ENEMY_TYPE_ADDRESSES = [0x0016, 0x0017, 0x0018, 0x0019, 0x001A]

# enemies whose context indicate that a stage change will occur (opposed to an
# enemy that implies a stage change wont occur -- i.e., a vine)
_STAGE_OVER_ENEMIES = np.array([0x2D, 0x31])
Shape = namedtuple('Shape', ['width', 'height'])
Point = namedtuple('Point', ['x', 'y'])


class Tile(object):
    __slots__ = ['type']

    def __init__(self, type: Enum):
        self.type = type


class Enemy(object):
    def __init__(self, enemy_id: int, location: Point, tile_location: Point):
        self.type = EnemyType(enemy_id)
        self.location = location
        self.tile_location = tile_location


def get_tile(x, y, ram, group_non_zero_tiles=False):
    page = (x // 256) % 2
    sub_x = (x % 256) // 16
    sub_y = (y - 32) // 16

    if sub_y not in range(13):
        return StaticType.Empty.value

    addr = 0x500 + page * 208 + sub_y * 16 + sub_x
    if group_non_zero_tiles:
        if ram[addr] != 0:
            return StaticType.Fake.value

    return ram[addr]


class SuperMarioBrosEnv(NESEnv):
    """An environment for playing Super Mario Bros with OpenAI Gym."""
    # the legal range of rewards for each step
    reward_range = (-15, 15)

    def __init__(self, rom_mode='vanilla', lost_levels=False, target=None):
        """
        Initialize a new Super Mario Bros environment.

        Args:
            rom_mode (str): the ROM mode to use when loading ROMs from disk
            lost_levels (bool): whether to load the ROM with lost levels.
                - False: load original Super Mario Bros.
                - True: load Super Mario Bros. Lost Levels
            target (tuple): a tuple of the (world, stage) to play as a level

        Returns:
            None

        """
        self.MAX_NUM_ENEMIES = 5
        self.PAGE_SIZE = 256
        self.NUM_BLOCKS = 8
        self.RESOLUTION = Shape(256, 240)
        self.NUM_TILES = 416  # 0x69f - 0x500 + 1
        self.NUM_SCREEN_PAGES = 2
        self.TOTAL_RAM = self.NUM_BLOCKS * self.PAGE_SIZE

        self.sprite = Shape(width=16, height=16)
        self.resolution = Shape(256, 240)
        self.status_bar = Shape(width=self.resolution.width, height=2 * self.sprite.height)

        self.xbins = list(range(16, self.resolution.width, 16))
        self.ybins = list(range(16, self.resolution.height, 16))
        # decode the ROM path based on mode and lost levels flag
        rom = rom_path(lost_levels, rom_mode)
        # initialize the super object with the ROM path
        super(SuperMarioBrosEnv, self).__init__(rom)
        # set the target world, stage, and area variables
        target = decode_target(target, lost_levels)
        self._target_world, self._target_stage, self._target_area = target
        # setup a variable to keep track of the last frames time
        self._time_last = 0
        # setup a variable to keep track of the last frames x position
        self._x_position_last = 0
        # reset the emulator
        self.reset()
        # skip the start screen
        self._skip_start_screen()
        # create a backup state to restore from on subsequent calls to reset
        self._backup()

    def get_enemy_locations(self):
        '''
            @Chrispresso -
        '''
        # We only care about enemies that are drawn. Others may exist
        # in memory, but if they aren't on the screen, they can't hurt us.
        # enemies = [None for _ in range(self.MAX_NUM_ENEMIES)]
        enemies = []

        for enemy_num in range(self.MAX_NUM_ENEMIES):
            enemy = self.ram[RAMLocations.Enemy_Drawn.value + enemy_num]
            # Is there an enemy? 1/0
            if enemy:
                # Get the enemy X location.
                x_pos_level = self.ram[RAMLocations.Enemy_X_Position_In_Level.value + enemy_num]
                x_pos_screen = self.ram[RAMLocations.Enemy_X_Position_On_Screen.value + enemy_num]
                enemy_loc_x = (x_pos_level * 0x100) + x_pos_screen  # - ram[0x71c]
                # print(ram[0x71c])
                # enemy_loc_x = ram[self.RAMLocations.Enemy_X_Position_Screen_Offset.value + enemy_num]
                # Get the enemy Y location.
                enemy_loc_y = self.ram[RAMLocations.Enemy_Y_Position_On_Screen.value + enemy_num]
                # Set location
                location = Point(enemy_loc_x, enemy_loc_y)
                ybin = np.digitize(enemy_loc_y, self.ybins)
                xbin = np.digitize(enemy_loc_x, self.xbins)
                tile_location = Point(xbin, ybin)

                # Grab the id
                enemy_id = self.ram[RAMLocations.Enemy_Type.value + enemy_num]
                # Create enemy-
                e = Enemy(enemy_id, location, tile_location)

                enemies.append(e)

        return enemies

    def get_powerup_locations(self):
        '''
        @Tazzioli Davide
        '''
        tile_location = None
        powerup = self.ram[RAMLocations.Powerup_Drawn.value]

        if powerup:
            x_pos_screen = self.ram[RAMLocations.Powerup_X_Position_On_Screen.value]
            y_pos_screen = self.ram[RAMLocations.Powerup_Y_Position_On_Screen.value]

            ybin = np.digitize(y_pos_screen, self.ybins)
            xbin = np.digitize(x_pos_screen, self.xbins)
            tile_location = Point(xbin, ybin)

        return tile_location

    def get_mario_location_in_level(self) -> Point:
        mario_x = self.ram[RAMLocations.Player_X_Postion_In_Level.value] * 256 + self.ram[
            RAMLocations.Player_X_Position_On_Screen.value]
        mario_y = self.ram[RAMLocations.Player_Y_Position_Screen_Offset.value]
        return Point(mario_x, mario_y)

    def get_mario_score(self) -> int:
        multiplier = 10
        score = 0
        for loc in range(0x07DC, 0x07D7 - 1, -1):
            score += self.ram[loc] * multiplier
            multiplier *= 10

        return score

    def get_mario_location_on_screen(self):
        mario_x = self.ram[RAMLocations.Player_X_Position_Screen_Offset.value]
        mario_y = self.ram[RAMLocations.Player_Y_Pos_On_Screen.value] * self.ram[
            RAMLocations.Player_Vertical_Screen_Position.value] + self.sprite.height
        return Point(mario_x, mario_y)

    def get_tile_type(self, delta_x: int, delta_y: int, mario: Point):
        x = mario.x + delta_x
        y = mario.y + delta_y + self.sprite.height

        # Tile locations have two pages. Determine which page we are in
        page = (x // 256) % 2
        # Figure out where in the page we are
        sub_page_x = (x % 256) // 16
        sub_page_y = (y - 32) // 16  # The PPU is not part of the world, coins, etc (status bar at top)
        if sub_page_y not in range(13):  # or sub_page_x not in range(16):
            return StaticType.Empty.value

        addr = 0x500 + page * 208 + sub_page_y * 16 + sub_page_x
        return self.ram[addr]

    def get_tile_loc(self, x, y):
        row = np.digitize(y, self.ybins) - 2
        col = np.digitize(x, self.xbins)
        return row, col

    @property
    def get_tiles(self):
        tiles = {}
        row = 0
        col = 0
        self.get_powerup_locations()
        mario_level = self.get_mario_location_in_level()
        mario_screen = self.get_mario_location_on_screen()
        x_start = mario_level.x - mario_screen.x
        enemies = self.get_enemy_locations()
        y_start = 0
        mx, my = self.get_mario_location_in_level()
        my += 16

        for y_pos in range(y_start, 240, 16):
            for x_pos in range(x_start, x_start + 256, 16):
                loc = (row, col)
                tile = get_tile(x_pos, y_pos, self.ram)
                x, y = x_pos, y_pos
                page = (x // 256) % 2
                sub_x = (x % 256) // 16
                sub_y = (y - 32) // 16
                #addr = 0x500 + page * 208 + sub_y * 16 + sub_x

                # row 0 and 1: game information -> irrelevant
                if row < 2:
                    tiles[loc] = StaticType.Empty
                elif StaticType.has_value(tile):
                    tiles[loc] = StaticType(tile)
                else:
                    # unrecognized tile -> became Ground
                    tiles[loc] = StaticType.Ground

                    # TODO check what happen with dynamic tiles (all unrecognized)

                for enemy in enemies:
                    ex = enemy.location.x
                    ey = enemy.location.y + 8
                    # Since we can only discriminate within 8 pixels, if it falls within this bound, count it as there
                    if abs(x_pos - ex) <= 8 and abs(y_pos - ey) <= 8:
                        tiles[loc] = enemy.type
                # Next col
                col += 1
            # Move to next row
            col = 0
            row += 1
        # Place marker for powerup
        powerup = self.get_powerup_locations()
        if powerup is not None:
            tiles[powerup] = StaticType.PowerUp
        # Place marker for mario
        mario_row, mario_col = self.get_mario_row_col()
        loc = (mario_row, mario_col)
        tiles[loc] = DynamicType.Mario

        return tiles

    def get_mario_row_col(self):
        x, y = self.get_mario_location_on_screen()
        # Adjust 16 for PPU
        y = self.ram[RAMLocations.Player_Y_Position_Screen_Offset.value] + 16
        x += 12
        col = x // 16
        row = (y - 0) // 16
        return row, col

    def _get_info(self):
        """Return the info after a step occurs"""
        return dict(
            coins=self._coins,
            flag_get=self._flag_get,
            life=self._life,
            score=self._score,
            stage=self._stage,
            status=self._player_status,
            time=self._time,
            world=self._world,
            x_pos=self._x_position,
            y_pos=self._y_position,
            tiles=self.get_tiles,
            mario_location=self.get_mario_row_col(),
            is_dead=self._is_dying
        )

    # Utility functions by @Christian Kauten - https://github.com/Kautenja/gym-super-mario-bros

    @property
    def is_single_stage_env(self):
        """Return True if this environment is a stage environment."""
        return self._target_world is not None and self._target_area is not None

    # MARK: Memory access

    def _read_mem_range(self, address, length):
        """
        Read a range of bytes where each byte is a 10's place figure.

        Args:
            address (int): the address to read from as a 16 bit integer
            length: the number of sequential bytes to read

        Note:
            this method is specific to Mario where three GUI values are stored
            in independent memory slots to save processing time
            - score has 6 10's places
            - coins has 2 10's places
            - time has 3 10's places

        Returns:
            the integer value of this 10's place representation

        """
        return int(''.join(map(str, self.ram[address:address + length])))

    @property
    def _level(self):
        """Return the level of the game."""
        return self.ram[0x075f] * 4 + self.ram[0x075c]

    @property
    def _world(self):
        """Return the current world (1 to 8)."""
        return self.ram[0x075f] + 1

    @property
    def _stage(self):
        """Return the current stage (1 to 4)."""
        return self.ram[0x075c] + 1

    @property
    def _area(self):
        """Return the current area number (1 to 5)."""
        return self.ram[0x0760] + 1

    @property
    def _score(self):
        """Return the current player score (0 to 999990)."""
        # score is represented as a figure with 6 10's places
        return self._read_mem_range(0x07de, 6)

    @property
    def _time(self):
        """Return the time left (0 to 999)."""
        # time is represented as a figure with 3 10's places
        return self._read_mem_range(0x07f8, 3)

    @property
    def _coins(self):
        """Return the number of coins collected (0 to 99)."""
        # coins are represented as a figure with 2 10's places
        return self._read_mem_range(0x07ed, 2)

    @property
    def _life(self):
        """Return the number of remaining lives."""
        return self.ram[0x075a]

    @property
    def _x_position(self):
        """Return the current horizontal position."""
        # add the current page 0x6d to the current x
        return self.ram[0x6d] * 0x100 + self.ram[0x86]

    @property
    def _left_x_position(self):
        """Return the number of pixels from the left of the screen."""
        # TODO: resolve RuntimeWarning: overflow encountered in ubyte_scalars
        # subtract the left x position 0x071c from the current x 0x86
        # return (self.ram[0x86] - self.ram[0x071c]) % 256
        return np.uint8(int(self.ram[0x86]) - int(self.ram[0x071c])) % 256

    @property
    def _y_pixel(self):
        """Return the current vertical position."""
        return self.ram[0x03b8]

    @property
    def _y_viewport(self):
        """
        Return the current y viewport.

        Note:
            1 = in visible viewport
            0 = above viewport
            > 1 below viewport (i.e. dead, falling down a hole)
            up to 5 indicates falling into a hole

        """
        return self.ram[0x00b5]

    @property
    def _y_position(self):
        """Return the current vertical position."""
        # check if Mario is above the viewport (the score board area)
        if self._y_viewport < 1:
            # y position overflows so we start from 255 and add the offset
            return 255 + (255 - self._y_pixel)
        # invert the y pixel into the distance from the bottom of the screen
        return 255 - self._y_pixel

    @property
    def _player_status(self):
        """Return the player status as a string."""
        return _STATUS_MAP[self.ram[0x0756]]

    @property
    def _player_state(self):
        """
        Return the current player state.

        Note:
            0x00 : Leftmost of screen
            0x01 : Climbing vine
            0x02 : Entering reversed-L pipe
            0x03 : Going down a pipe
            0x04 : Auto-walk
            0x05 : Auto-walk
            0x06 : Dead
            0x07 : Entering area
            0x08 : Normal
            0x09 : Cannot move
            0x0B : Dying
            0x0C : Palette cycling, can't move

        """
        return self.ram[0x000e]

    @property
    def _is_dying(self):
        """Return True if Mario is in dying animation, False otherwise."""
        return self._player_state == 0x0b or self._y_viewport > 1

    @property
    def _is_dead(self):
        """Return True if Mario is dead, False otherwise."""
        return self._player_state == 0x06

    @property
    def _is_game_over(self):
        """Return True if the game has ended, False otherwise."""
        # the life counter will get set to 255 (0xff) when there are no lives
        # left. It goes 2, 1, 0 for the 3 lives of the game
        return self._life == 0xff

    @property
    def _is_busy(self):
        """Return boolean whether Mario is busy with in-game garbage."""
        return self._player_state in _BUSY_STATES

    @property
    def _is_world_over(self):
        """Return a boolean determining if the world is over."""
        # 0x0770 contains GamePlay mode:
        # 0 => Demo
        # 1 => Standard
        # 2 => End of world
        return self.ram[0x0770] == 2

    @property
    def _is_stage_over(self):
        """Return a boolean determining if the level is over."""
        # iterate over the memory addresses that hold enemy types
        for address in _ENEMY_TYPE_ADDRESSES:
            # check if the byte is either Bowser (0x2D) or a flag (0x31)
            # this is to prevent returning true when Mario is using a vine
            # which will set the byte at 0x001D to 3
            if self.ram[address] in _STAGE_OVER_ENEMIES:
                # player float state set to 3 when sliding down flag pole
                return self.ram[0x001D] == 3

        return False

    @property
    def _flag_get(self):
        """Return a boolean determining if the agent reached a flag."""
        return self._is_world_over or self._is_stage_over

    # MARK: RAM Hacks

    def _write_stage(self):
        """Write the stage data to RAM to overwrite loading the next stage."""
        self.ram[0x075f] = self._target_world - 1
        self.ram[0x075c] = self._target_stage - 1
        self.ram[0x0760] = self._target_area - 1

    def _runout_prelevel_timer(self):
        """Force the pre-level timer to 0 to skip frames during a death."""
        self.ram[0x07A0] = 0

    def _skip_change_area(self):
        """Skip change area animations by by running down timers."""
        change_area_timer = self.ram[0x06DE]
        if 1 < change_area_timer < 255:
            self.ram[0x06DE] = 1

    def _skip_occupied_states(self):
        """Skip occupied states by running out a timer and skipping frames."""
        while self._is_busy or self._is_world_over:
            self._runout_prelevel_timer()
            self._frame_advance(0)

    def _skip_start_screen(self):
        """Press and release start to skip the start screen."""
        # press and release the start button
        self._frame_advance(8)
        self._frame_advance(0)
        # Press start until the game starts
        while self._time == 0:
            # press and release the start button
            self._frame_advance(8)
            # if we're in the single stage, environment, write the stage data
            if self.is_single_stage_env:
                self._write_stage()
            self._frame_advance(0)
            # run-out the prelevel timer to skip the animation
            self._runout_prelevel_timer()
        # set the last time to now
        self._time_last = self._time
        # after the start screen idle to skip some extra frames
        while self._time >= self._time_last:
            self._time_last = self._time
            self._frame_advance(8)
            self._frame_advance(0)

    def _skip_end_of_world(self):
        """Skip the cutscene that plays at the end of a world."""
        if self._is_world_over:
            # get the current game time to reference
            time = self._time
            # loop until the time is different
            while self._time == time:
                # frame advance with NOP
                self._frame_advance(0)

    def _kill_mario(self):
        """Skip a death animation by forcing Mario to death."""
        # force Mario's state to dead
        self.ram[0x000e] = 0x06
        # step forward one frame
        self._frame_advance(0)

    # MARK: Reward Function

    @property
    def _x_reward(self):
        """Return the reward based on left right movement between steps."""
        _reward = self._x_position - self._x_position_last
        self._x_position_last = self._x_position
        # TODO: check whether this is still necessary
        # resolve an issue where after death the x position resets. The x delta
        # is typically has at most magnitude of 3, 5 is a safe bound
        if _reward < -5 or _reward > 5:
            return 0

        return _reward

    @property
    def _time_penalty(self):
        """Return the reward for the in-game clock ticking."""
        _reward = self._time - self._time_last
        self._time_last = self._time
        # time can only decrease, a positive reward results from a reset and
        # should default to 0 reward
        if _reward > 0:
            return 0

        return _reward

    @property
    def _death_penalty(self):
        """Return the reward earned by dying."""
        if self._is_dying or self._is_dead:
            return -25

        return 0

    # MARK: nes-py API calls

    def _will_reset(self):
        """Handle and RAM hacking before a reset occurs."""
        self._time_last = 0
        self._x_position_last = 0

    def _did_reset(self):
        """Handle any RAM hacking after a reset occurs."""
        self._time_last = self._time
        self._x_position_last = self._x_position

    def _did_step(self, done):
        """
        Handle any RAM hacking after a step occurs.

        Args:
            done: whether the done flag is set to true

        Returns:
            None

        """
        # if done flag is set a reset is incoming anyway, ignore any hacking
        if done:
            return
        # if mario is dying, then cut to the chase and kill hi,
        if self._is_dying:
            self._kill_mario()
        # skip world change scenes (must call before other skip methods)
        if not self.is_single_stage_env:
            self._skip_end_of_world()
        # skip area change (i.e. enter pipe, flag get, etc.)
        self._skip_change_area()
        # skip occupied states like the black screen between lives that shows
        # how many lives the player has left
        self._skip_occupied_states()

    def _get_reward(self):
        """Return the reward after a step occurs."""
        return self._x_reward + self._time_penalty + self._death_penalty

    def _get_done(self):
        """Return True if the episode is over, False otherwise."""
        if self.is_single_stage_env:
            return self._is_dying or self._is_dead or self._flag_get
        return self._is_game_over

# explicitly define the outward facing API of this module
__all__ = [SuperMarioBrosEnv.__name__]
