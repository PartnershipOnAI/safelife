"""
Utilities to graphically render SafeLifeGame boards.
"""

import os
import imageio
import numpy as np

from .safelife_game import CellTypes, GameState
from .helper_utils import recenter_view


sprite_path = os.path.join(os.path.dirname(__file__), "sprites.png")
sprite_sheet = imageio.imread(os.path.abspath(sprite_path)) / 255
SPRITE_SIZE = 14


def load_sprite(i, j):
    s = SPRITE_SIZE
    return sprite_sheet[s*i:s*i+s, s*j:s*j+s]


sprites = {
    CellTypes.agent: [
        load_sprite(0, 1),
        load_sprite(0, 2),
        load_sprite(0, 3),
        load_sprite(0, 4),
    ],
    CellTypes.empty: load_sprite(0, 0),
    CellTypes.life: load_sprite(1, 0),
    CellTypes.alive: load_sprite(1, 1),
    CellTypes.wall: load_sprite(2, 2),
    CellTypes.crate: load_sprite(2, 3),
    CellTypes.plant: load_sprite(1, 3),
    CellTypes.tree: load_sprite(1, 4),
    CellTypes.ice_cube: load_sprite(2, 0),
    CellTypes.parasite: load_sprite(2, 4),
    CellTypes.weed: load_sprite(1, 2),
    CellTypes.spawner: load_sprite(3, 0),
    CellTypes.hard_spawner: load_sprite(3, 2),
    CellTypes.level_exit: load_sprite(3, 1),
    CellTypes.fountain: load_sprite(2, 1),
}

foreground_colors = np.array([
    [0.4, 0.4, 0.4],  # black
    [0.8, 0.2, 0.2],  # red
    [0.2, 0.8, 0.2],  # green
    [0.8, 0.8, 0.2],  # yellow
    [0.2, 0.2, 0.8],  # blue
    [0.8, 0.2, 0.8],  # magenta
    [0.2, 0.8, 0.8],  # cyan
    [1.0, 1.0, 1.0],  # white
])

background_colors = np.array([
    [0.6, 0.6, 0.6],  # black
    [0.9, 0.6, 0.6],  # red
    [0.6, 0.9, 0.6],  # green
    [0.9, 0.9, 0.6],  # yellow
    [0.5, 0.5, 0.9],  # blue
    [0.9, 0.6, 0.9],  # magenta
    [0.6, 0.9, 0.9],  # cyan
    [0.9, 0.9, 0.9],  # white
])

cell_array = np.array([
    # [CellTypes.empty, (5*0 + 0) + 1],
    [CellTypes.life, (5*1 + 0) + 1],
    [CellTypes.alive, (5*1 + 1) + 1],
    [CellTypes.wall, (5*2 + 2) + 1],
    [CellTypes.crate, (5*2 + 3) + 1],
    [CellTypes.plant, (5*1 + 3) + 1],
    [CellTypes.tree, (5*1 + 4) + 1],
    [CellTypes.ice_cube, (5*2 + 0) + 1],
    [CellTypes.parasite, (5*2 + 4) + 1],
    [CellTypes.weed, (5*1 + 2) + 1],
    [CellTypes.spawner, (5*3 + 0) + 1],
    [CellTypes.hard_spawner, (5*3 + 2) + 1],
    [CellTypes.level_exit, (5*3 + 1) + 1],
    [CellTypes.fountain, (5*2 + 1) + 1],
])
sprites_array = np.array([load_sprite(n // 5, n % 5) for n in range(20)])


def render_board(board, goals, orientation, edit_loc=None, edit_color=0):
    if edit_loc and (edit_loc[0] >= board.shape[0] or edit_loc[1] >= board.shape[1]):
        edit_loc = None

    agent_idx = ((board & CellTypes.agent) > 0) * (2 + orientation)
    sprite_idx = -1 + np.sum(
        ((board[...,None] & ~CellTypes.rainbow_color) == cell_array[:,0])
        * cell_array[:,1], axis=-1)
    sprite_idx *= board > 0
    sprite_idx += agent_idx
    fg_color = foreground_colors[(board & CellTypes.rainbow_color) >> 9]
    bg_color = background_colors[(goals & CellTypes.rainbow_color) >> 9]
    sprites = sprites_array[sprite_idx]
    mask, sprite = sprites[...,3:], sprites[...,:3]
    tile = (1-mask) * bg_color[...,None,None,:]
    tile += mask * sprite * fg_color[...,None,None,:]
    data = (255 * tile).astype(np.uint8)
    if edit_loc is not None:
        # should do some error checking to make sure this is in range
        edit_cell = data[edit_loc[1], edit_loc[0]]
        edit_cell[[0,1,-1,-2]] = edit_color
        edit_cell[:,[0,1,-1,-2]] = edit_color
    data = np.moveaxis(data, -4, -3)
    s = data.shape
    data = data.reshape(s[:-5] + (s[-5]*s[-4], s[-3]*s[-2], s[-1]))
    return data


def render_game(game, view_size=None, edit_mode=None):
    """
    Render the game as a numpy rgb array.

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

    Returns
    -------
    numpy array
        Has shape (view_size) + (3,).
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
    edit_color = foreground_colors[
        (game.edit_color & CellTypes.rainbow_color) >> 9] * 255
    if edit_mode == "GOALS":
        # Render goals instead. Swap board and goals.
        board, goals = goals, board
    return render_board(board, goals, game.orientation, edit_loc, edit_color)


def render_file(fname, fps=30, data=None):
    """
    Load a saved SafeLifeGame file and render it as a png or gif.

    The game will be rendered as an animated gif if it contains a sequence of
    states; otherwise it will be rendered as a png.

    Parameters
    ----------
    fname : str
    fps : float
        Frames per second for gif animation.
    """
    bare_fname = '.'.join(fname.split('.')[:-1])
    if data is None:
        data = np.load(fname)

    if hasattr(data, 'keys') and 'levels' in data:
        os.makedirs(bare_fname, exist_ok=True)
        for level in data['levels']:
            render_file(os.path.join(bare_fname, level['name']), fps, level)
        return

    rgb_array = render_board(
        data['board'], data['goals'], data['orientation'][..., None, None])
    if rgb_array.ndim == 3:
        imageio.imwrite(bare_fname+'.png', rgb_array)
    elif rgb_array.ndim == 4:
        imageio.mimwrite(bare_fname+'.gif', rgb_array,
                         duration=1/fps, subrectangles=True)
    else:
        raise Exception("Unexpected dimension of rgb_array.")


def render_mov(fname, steps, fps=30):
    """
    Load a saved SafeLifeGame state and render it as an animated gif.

    Parameters
    ----------
    fname : str
    steps : int
        The number of steps to evolve the game state. This is the same
        as the number of frames that will be rendered.
    fps : float
        Frames per second for gif animation.
    """
    game = GameState.load(fname)
    bare_fname = '.'.join(fname.split('.')[:-1])
    frames = []
    for _ in range(steps):
        frames.append(render_game(game))
        game.advance_board()
    imageio.mimwrite(bare_fname+'.gif', frames,
                     duration=1/fps, subrectangles=True)


def _make_cmd_args(subparsers):
    # used by __main__.py to define command line tools
    from argparse import RawDescriptionHelpFormatter
    import textwrap
    parser = subparsers.add_parser(
        "render", help="Convert a SafeLife level to either a png or a gif.",
        description=textwrap.dedent("""
        Convert a SafeLife level to either a png or a gif.

        Static SafeLife levels can be saved while editing them during
        interactive play, and an agent's actions can be saved either during
        training or while recording interactive play. Either way, the data
        will be saved in .npz files. Static files will get rendered to png,
        while recorded actions will be compiled into an animated gif.
        """), formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument('fnames', help="File(s) to render.", nargs='+')
    parser.add_argument('--steps', default=0, type=int,
        help="Static output can be turned into animated output by setting"
        " this to be non-zero. If non-zero, it determines the number of"
        " steps that the board animation runs during animation, with one"
        " step for each frame.")
    parser.add_argument('--fps', default=30, type=float,
        help="Frames per second for animated outputs.")
    parser.set_defaults(run_cmd=_run_cmd_args)


def _run_cmd_args(args):
    for fname in args.fnames:
        try:
            if args.steps == 0:
                render_file(fname, args.fps)
            else:
                render_mov(fname, args.steps, args.fps)
            print("Success:", fname)
        except Exception:
            print("Failed:", fname)
