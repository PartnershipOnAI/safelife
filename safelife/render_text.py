import numpy as np

from .helper_utils import recenter_view
from .safelife_game import CellTypes, GameWithGoals


background_colors = [
    '\x1b[48;5;251m',  # black / empty
    '\x1b[48;5;217m',  # red
    '\x1b[48;5;114m',  # green
    '\x1b[48;5;229m',  # yellow
    '\x1b[48;5;117m',  # blue
    '\x1b[48;5;183m',  # magenta
    '\x1b[48;5;123m',  # cyan
    '\x1b[48;5;255m',  # white
]

foreground_colors = [
    '\x1b[38;5;0m',  # black
    '\x1b[38;5;1m',  # red
    '\x1b[38;5;2m',  # green
    '\x1b[38;5;172m',  # yellow
    '\x1b[38;5;12m',  # blue
    '\x1b[38;5;129m',  # magenta
    '\x1b[38;5;39m',  # cyan
    '\x1b[38;5;244m',  # white / gray
]


def print_reward_table():
    text = ""
    rewards = GameWithGoals.reward_table
    for r in range(8):
        text += background_colors[r]
        for c in range(8):
            text += foreground_colors[c]
            text += "{:2d} ".format(rewards[r,c])
        text += '\x1b[0m\n'
    print(text)


@np.vectorize
def render_cell(cell, goal=0, orientation=0, edit_color=None):
    cell_color = (cell & CellTypes.rainbow_color) >> CellTypes.color_bit
    goal_color = (goal & CellTypes.rainbow_color) >> CellTypes.color_bit
    val = background_colors[goal_color]
    val += ' ' if edit_color is None else foreground_colors[edit_color] + '∎'
    val += foreground_colors[cell_color]

    if cell & CellTypes.agent:
        arrow = '⋀>⋁<'[orientation]
        val += '\x1b[1m' + arrow
    else:
        gray_cell = cell & ~CellTypes.rainbow_color
        val += {
            CellTypes.empty: '.' if cell_color else ' ',
            CellTypes.life: 'z',
            CellTypes.alive: 'Z',
            CellTypes.wall: '#',
            CellTypes.crate: '%',
            CellTypes.plant: '&',
            CellTypes.tree: 'T',
            CellTypes.ice_cube: '=',
            CellTypes.parasite: '!',
            CellTypes.weed: '@',
            CellTypes.spawner: 's',
            CellTypes.hard_spawner: 'S',
            CellTypes.level_exit: 'X',
            CellTypes.fountain: '\x1b[1m+',
        }.get(gray_cell, '?')
    return val + '\x1b[0m'


def cell_name(cell):
    cell_type = {
        CellTypes.empty: 'empty',
        CellTypes.life: 'life',
        CellTypes.alive: 'hard-life',
        CellTypes.wall: 'wall',
        CellTypes.crate: 'crate',
        CellTypes.plant: 'plant',
        CellTypes.tree: 'tree',
        CellTypes.ice_cube: 'ice-cube',
        CellTypes.parasite: 'parasite',
        CellTypes.weed: 'weed',
        CellTypes.spawner: 'spawner',
        CellTypes.hard_spawner: 'hard-spawner',
        CellTypes.level_exit: 'exit',
        CellTypes.fountain: 'fountain',
    }.get(cell & ~CellTypes.rainbow_color, 'unknown')
    color = {
        0: 'gray',
        CellTypes.color_r: 'red',
        CellTypes.color_g: 'green',
        CellTypes.color_b: 'blue',
        CellTypes.color_r | CellTypes.color_b: 'magenta',
        CellTypes.color_g | CellTypes.color_r: 'yellow',
        CellTypes.color_b | CellTypes.color_g: 'cyan',
        CellTypes.rainbow_color: 'white',
    }.get(cell & CellTypes.rainbow_color, 'x')
    return cell_type + '-' + color


def render_board(board, goals=0, orientation=0, edit_loc=None, edit_color=0):
    """
    Just render the board itself. Doesn't require game state.
    """
    if edit_loc and (edit_loc[0] >= board.shape[0] or edit_loc[1] >= board.shape[1]):
        edit_loc = None
    goals = np.broadcast_to(goals, board.shape)

    screen = np.empty((board.shape[0]+2, board.shape[1]+3,), dtype=object)
    screen[:] = ''
    screen[0] = screen[-1] = ' -'
    screen[:,0] = screen[:,-2] = ' |'
    screen[:,-1] = '\n'
    screen[0,0] = screen[0,-2] = screen[-1,0] = screen[-1,-2] = ' +'
    screen[1:-1,1:-2] = render_cell(board, goals, orientation)

    if edit_loc:
        x1, y1 = edit_loc
        val = render_cell(board[y1, x1], goals[y1, x1], orientation, edit_color)
        screen[y1+1, x1+1] = str(val)
    return ''.join(screen.ravel())


def render_game(game, view_size=None, edit_mode=None):
    """
    Render the game as an ansi string.

    Parameters
    ----------
    game : SafeLifeGame instance
    view_size : (int, int) or None
        Shape of the view port, or None if the full board should be rendered.
        If not None, the view will be centered on either the agent or the
        current edit location.
    edit_mode : None, "BOARD", or "GOALS"
        Determines whether or not the game should be drawn in edit mode with
        the edit cursor. If "GOALS", the goals and normal board are swapped so
        that the goals can be edited directly.
    """
    if view_size is not None:
        if edit_mode:
            center = game.edit_loc
            edit_loc = view_size[1] // 2, view_size[0] // 2
        else:
            center = game.agent_loc
            edit_loc = None
        center = game.edit_loc if edit_mode else game.agent_loc
        board = recenter_view(game.board, view_size, center[::-1], game.exit_locs)
        goals = recenter_view(game.goals, view_size, center[::-1])
    else:
        board = game.board
        goals = game.goals
        edit_loc = game.edit_loc if edit_mode else None
    edit_color = (game.edit_color & CellTypes.rainbow_color) >> CellTypes.color_bit
    if edit_mode == "GOALS":
        # Render goals instead. Swap board and goals.
        board, goals = goals, board

    return render_board(board, goals, game.orientation, edit_loc, edit_color)


def agent_powers(game):
    x0, y0 = game.agent_loc
    agent = game.board[y0, x0]
    power_names = [
        (CellTypes.alive, 'alive'),
        (CellTypes.preserving, 'preserving'),
        (CellTypes.inhibiting, 'inhibiting'),
        (CellTypes.spawning, 'spawning'),
    ]
    powers = [txt for val, txt in power_names if agent & val]
    return ', '.join(powers) or 'none'


if __name__ == "__main__":
    print_reward_table()
