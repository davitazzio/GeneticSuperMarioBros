from enum import Enum, unique


@unique
class RAMLocations(Enum):
    # Since the max number of enemies on the screen is 5, the addresses for enemies are
    # the starting address and span a total of 5 bytes. This means Enemy_Drawn + 0 is the
    # whether or not enemy 0 is drawn, Enemy_Drawn + 1 is enemy 1, etc. etc.
    Enemy_Drawn = 0x0F
    Enemy_Type = 0x16
    Enemy_X_Position_In_Level = 0x6E
    Enemy_X_Position_On_Screen = 0x87
    Enemy_Y_Position_On_Screen = 0xCF

    Player_X_Postion_In_Level = 0x06D
    Player_X_Position_On_Screen = 0x086

    Powerup_Drawn = 0x0014
    Powerup_X_Position_On_Screen = 0x008C
    Powerup_Y_Position_On_Screen = 0x00D4

    Player_X_Position_Screen_Offset = 0x3AD
    Player_Y_Position_Screen_Offset = 0x3B8
    Enemy_X_Position_Screen_Offset = 0x3AE

    Player_Y_Pos_On_Screen = 0xCE
    Player_Vertical_Screen_Position = 0xB5

@unique
class EnemyType(Enum):
    Green_Koopa1 = 0x00
    Red_Koopa1 = 0x01
    Buzzy_Beetle = 0x02
    Red_Koopa2 = 0x03
    Green_Koopa2 = 0x04
    Hammer_Brother = 0x05
    Goomba = 0x06
    Blooper = 0x07
    Bullet_Bill = 0x08
    Green_Koopa_Paratroopa = 0x09
    Grey_Cheep_Cheep = 0x0A
    Red_Cheep_Cheep = 0x0B
    Pobodoo = 0x0C
    Piranha_Plant = 0x0D
    Green_Paratroopa_Jump = 0x0E
    Bowser_Flame1 = 0x10
    Lakitu = 0x11
    Spiny_Egg = 0x12
    Fly_Cheep_Cheep = 0x14
    Bowser_Flame2 = 0x15


    # Generic_Enemy = 0xFF

    @classmethod
    def has_value(cls, value: int) -> bool:
        return value in set(item.value for item in cls)


@unique
class StaticType(Enum):
    Empty = 0x00
    Fake = 0x01
    Ground = 0x54
    Top_Pipe1 = 0x12
    Top_Pipe2 = 0x13
    Top_Pipe3 = 0x10
    Top_Pipe4 = 0x11
    Bottom_Pipe1 = 0x14
    Bottom_Pipe2 = 0x15
    Flagpole_Top = 0x24
    Flagpole = 0x25
    Coin_Block1 = 0xC0
    #Coin_Block2 = 0xC4
    Coin = 0xC2
    Breakable_Block = 0x51
    PowerUp_Block = 0xC1
    Hidden_life = 0x60
    Hit_animation_coinblock = 0x23
    Hitted_coinblock = 0xC4
    Hidden_Coin = 0x58
    Hidden_Powerup = 0x57
    PowerUp = -1
    Static_Block = 0x61

    # Generic_Static_Tile = 0xFF

    @classmethod
    def has_value(cls, value: int) -> bool:
        return value in set(item.value for item in cls)


@unique
class DynamicType(Enum):
    Mario = 0xAA

    Static_Lift1 = 0x24
    Static_Lift2 = 0x25
    Vertical_Lift1 = 0x26
    Vertical_Lift2 = 0x27
    Horizontal_Lift = 0x28
    Falling_Static_Lift = 0x29
    Horizontal_Moving_Lift = 0x2A
    Lift1 = 0x2B
    Lift2 = 0x2C
    Vine = 0x2F
    Flagpole = 0x30
    Start_Flag = 0x31
    Jump_Spring = 0x32
    Warpzone = 0x34
    Spring1 = 0x67
    Spring2 = 0x68
    PowerUp = 0x7E020C #0x0039

    # Generic_Dynamic_Tile = 0xFF

    @classmethod
    def has_value(cls, value: int) -> bool:
        return value in set(item.value for item in cls)
