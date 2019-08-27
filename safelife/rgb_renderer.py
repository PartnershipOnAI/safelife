import os
import imageio
import numpy as np

from .game_physics import CellTypes, GameState
from .helper_utils import recenter_view


sprite_sheet = imageio.imread(os.path.abspath(
    os.path.join(__file__, "../sprites.png"))) / 255
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
    CellTypes.predator: load_sprite(2, 4),
    CellTypes.weed: load_sprite(1, 2),
    CellTypes.spawner: load_sprite(3, 0),
    CellTypes.level_exit: load_sprite(3, 1),
    CellTypes.fountain: load_sprite(2, 1),
}

foreground_colors = {
    CellTypes.empty: np.array([0.4, 0.4, 0.4]),
    CellTypes.color_r: np.array([0.8, 0.2, 0.2]),
    CellTypes.color_g: np.array([0.2, 0.8, 0.2]),
    CellTypes.color_g | CellTypes.color_r: np.array([0.8, 0.8, 0.2]),
    CellTypes.color_b: np.array([0.2, 0.2, 0.8]),
    CellTypes.color_b | CellTypes.color_r: np.array([0.8, 0.2, 0.8]),
    CellTypes.color_b | CellTypes.color_g: np.array([0.2, 0.8, 0.8]),
    CellTypes.rainbow_color: np.array([1.0, 1.0, 1.0])
}

background_colors = {
    CellTypes.empty: np.array([0.6, 0.6, 0.6]),
    CellTypes.color_r: np.array([0.9, 0.6, 0.6]),
    CellTypes.color_g | CellTypes.color_r: np.array([0.9, 0.9, 0.6]),
    CellTypes.color_g: np.array([0.6, 0.9, 0.6]),
    CellTypes.color_b: np.array([0.5, 0.5, 0.9]),
    CellTypes.color_b | CellTypes.color_r: np.array([0.9, 0.6, 0.9]),
    CellTypes.color_b | CellTypes.color_g: np.array([0.6, 0.9, 0.9]),
    CellTypes.rainbow_color: np.array([0.9, 0.9, 0.9])
}

fg_array = np.array(list(foreground_colors.values()))
bg_array = np.array(list(background_colors.values()))
cell_array = np.array([
    [CellTypes.empty, (5*0 + 0) + 1],
    [CellTypes.life, (5*1 + 0) + 1],
    [CellTypes.alive, (5*1 + 1) + 1],
    [CellTypes.wall, (5*2 + 2) + 1],
    [CellTypes.crate, (5*2 + 3) + 1],
    [CellTypes.plant, (5*1 + 3) + 1],
    [CellTypes.tree, (5*1 + 4) + 1],
    [CellTypes.ice_cube, (5*2 + 0) + 1],
    [CellTypes.predator, (5*2 + 4) + 1],
    [CellTypes.weed, (5*1 + 2) + 1],
    [CellTypes.spawner, (5*3 + 0) + 1],
    [CellTypes.level_exit, (5*3 + 1) + 1],
    [CellTypes.fountain, (5*2 + 1) + 1],
])
sprites_array = np.array([load_sprite(n // 5, n % 5) for n in range(20)])


def render_cell(cell, goal=0, orientation=0):
    fg_color = foreground_colors[cell & CellTypes.rainbow_color]
    bg_color = background_colors[goal & CellTypes.rainbow_color]
    cell = cell & ~CellTypes.rainbow_color
    if cell & CellTypes.agent:
        sprite = sprites[CellTypes.agent][orientation]
    else:
        sprite = sprites.get(cell, sprites[0])
    mask, sprite = sprite[:,:,3:], sprite[:,:,:3]
    tile = (1-mask) * bg_color + mask * sprite * fg_color
    return (255 * tile).astype(np.uint8)


render_cell = np.vectorize(
    render_cell, signature="(),(),()->({s},{s},3)".format(s=SPRITE_SIZE))


def render_board(board, goals, orientation, edit_loc=None):
    agent_idx = ((board & CellTypes.agent) > 0) * (2 + orientation)
    sprite_idx = -1 + np.sum(
        ((board[...,None] & ~CellTypes.rainbow_color) == cell_array[:,0])
        * cell_array[:,1], axis=-1)
    sprite_idx += agent_idx
    fg_color = fg_array[(board & CellTypes.rainbow_color) >> 9]
    bg_color = bg_array[(goals & CellTypes.rainbow_color) >> 9]
    sprites = sprites_array[sprite_idx]
    mask, sprite = sprites[...,3:], sprites[...,:3]
    tile = (1-mask) * bg_color[...,None,None,:]
    tile += mask * sprite * fg_color[...,None,None,:]
    data = (255 * tile).astype(np.uint8)
    if edit_loc is not None:
        # should do some error checking to make sure this is in range
        edit_cell = data[edit_loc[1], edit_loc[0]]
        red = [255,0,0]
        edit_cell[0] = red
        edit_cell[-1] = red
        edit_cell[:,0] = red
        edit_cell[:,-1] = red
    data = np.moveaxis(data, -4, -3)
    s = data.shape
    data = data.reshape(s[:-5] + (s[-5]*s[-4], s[-3]*s[-2], s[-1]))
    return data


def render_game(game, view_size=None):
    if view_size is not None:
        if game.is_editing:
            center = game.edit_loc
            edit_loc = view_size[1] // 2, view_size[0] // 2
        else:
            center = game.agent_loc
            edit_loc = None
        center = game.edit_loc if game.is_editing else game.agent_loc
        board = recenter_view(game.board, view_size, center[::-1], game.exit_locs)
        goals = recenter_view(game.goals, view_size, center[::-1])
    else:
        board = game.board
        goals = game.goals
        edit_loc = game.edit_loc if game.is_editing else None
    return render_board(board, goals, game.orientation, edit_loc)


def render_file(fname, duration=0.03):
    data = np.load(fname)
    rgb_array = render_board(
        data['board'], data['goals'], data['orientation'][..., None, None])
    bare_fname = '.'.join(fname.split('.')[:-1])
    if rgb_array.ndim == 3:
        imageio.imwrite(bare_fname+'.png', rgb_array)
    elif rgb_array.ndim == 4:
        imageio.mimwrite(bare_fname+'.gif', rgb_array,
                         duration=duration, subrectangles=True)
    else:
        raise Exception("Unexpected dimension of rgb_array.")


def render_mov(fname, steps, duration=0.03):
    game = GameState.load(fname)
    bare_fname = '.'.join(fname.split('.')[:-1])
    frames = []
    for _ in range(steps):
        frames.append(render_game(game))
        game.advance_board()
    imageio.mimwrite(bare_fname+'.gif', frames,
                     duration=duration, subrectangles=True)


def _make_cmd_args(subparsers):
    # used by __main__.py to define command line tools
    parser = subparsers.add_parser(
        "render", help="Convert a SafeLife level to either a png or a gif.")
    parser.add_argument('fnames', help="File to render.", nargs='+')
    parser.add_argument('--steps', default=0, type=int)
    parser.add_argument('--duration', default=0.03, type=float)
    parser.set_defaults(run_cmd=_run_cmd_args)


def _run_cmd_args(args):
    for fname in args.fnames:
        try:
            if args.steps == 0:
                render_file(fname, args.duration)
            else:
                render_mov(fname, args.steps, args.duration)
            print("Success:", fname)
        except Exception:
            print("Failed:", fname)
