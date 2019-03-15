import numpy as np

from .array_utils import wrapping_array
from .game_physics import CellTypes, GameWithGoals


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
            text += f"{rewards[r,c]:2d} "
        text += '\x1b[0m\n'
    print(text)


@np.vectorize
def render_cell(cell, goal=0, orientation=0, edit_color=None):
    cell_color = (cell & CellTypes.rainbow_color) >> CellTypes.color_bit
    goal_color = (goal & CellTypes.rainbow_color) >> CellTypes.color_bit
    val = background_colors[goal_color]
    val += ' ' if edit_color is None else foreground_colors[edit_color] + '•'
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
            CellTypes.predator: '!',
            CellTypes.weed: '@',
            CellTypes.spawner: 'S',
            CellTypes.level_exit: 'X',
            CellTypes.fountain: '\x1b[1m+',
        }.get(gray_cell, '?')
    return val + '\x1b[0m'


def render_board(s, centered_view=False, view_size=None, fixed_orientation=False):
    """
    Renders the game state `s`. Does not include scores, etc.

    This is not exactly a speedy rendering system, but it should be plenty
    fast enough for our purposes.

    Parameters
    ----------
    view_size : (int width, int height)
        If not None, specifies the size of the view centered on the agent.
    fixed_orientation : bool
        If true, the board is re-oriented such that the player is always
        facing up.
    """
    if centered_view or view_size or fixed_orientation:
        if view_size is None:
            view_size = s.board.shape
        if fixed_orientation and s.orientation % 2 == 1:
            # transpose the view
            view_height, view_width = view_size
        else:
            view_width, view_height = view_size
        x0, y0 = s.agent_loc
        x0 -= view_width // 2
        y0 -= view_height // 2
        board = s.board.view(wrapping_array)[y0:y0+view_height, x0:x0+view_width]
        goals = s.goals.view(wrapping_array)[y0:y0+view_height, x0:x0+view_width]
    else:
        view_width, view_height = s.width, s.height
        board = s.board
        goals = s.goals
    screen = np.empty((view_height+2, view_width+3), dtype=object)
    screen[:] = ''
    screen[0] = screen[-1] = ' -'
    screen[:,0] = screen[:,-2] = ' |'
    screen[:,-1] = '\n'
    screen[0,0] = screen[0,-2] = screen[-1,0] = screen[-1,-2] = ' +'
    if fixed_orientation and s.orientation != 0:
        cells = render_cell(board, goals).view(np.ndarray)
        if s.orientation == 1:
            cells = cells.T[::-1]
        elif s.orientation == 2:
            cells = cells[::-1, ::-1]
        elif s.orientation == 3:
            cells = cells.T[:, ::-1]
        else:
            raise RuntimeError("Unexpected orientation: %s" % (s.orientation,))
        screen[1:-1,1:-2] = cells
    else:
        screen[1:-1,1:-2] = render_cell(board, goals, s.orientation)
    if s.is_editing:
        x0, y0 = s.agent_loc
        x1, y1 = s.edit_loc
        color = (board[y0, x0] & CellTypes.rainbow_color) >> CellTypes.color_bit
        val = render_cell(board[y1, x1], goals[y1, x1], s.orientation,
                          edit_color=color)
        screen[y1+1, x1+1] = str(val)
    return ''.join(screen.ravel())


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
