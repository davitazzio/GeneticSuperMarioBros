"""Static action sets for binary to discrete action space wrappers."""


# actions for the simple run right environment
RIGHT_ONLY = [
    ['NOOP'],
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
]

ALL_COMBINATION = [
    ['right'],
    ['left'],
    ['A'],
    ['B'],
    ['right', 'A'],
    ['left', 'A'],
    ['right', 'B'],
    ['left', 'B'],
    ['right', 'A', 'B'],

]


# actions for very simple movement
SIMPLE_MOVEMENT = [
    ['NOOP'],
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
    ['A'],
    ['left'],
]

JOYPAD = [
    ['up'],
    ['down'],
    ['left'],
    ['right'],
    ['A'],
    ['B'],
    ['NOOP']
]


# actions for more complex movement
COMPLEX_MOVEMENT = [
    ['NOOP'],
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
    ['A'],
    ['left'],
    ['left', 'A'],
    ['left', 'B'],
    ['left', 'A', 'B'],
    ['down'],
    ['up'],
]
