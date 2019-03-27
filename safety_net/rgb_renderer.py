import os
import imageio
import numpy as np

from .game_physics import CellTypes


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
    CellTypes.color_b: np.array([0.2, 0.2, 0.8]),
    CellTypes.color_r | CellTypes.color_g: np.array([0.8, 0.8, 0.2]),
    CellTypes.color_g | CellTypes.color_b: np.array([0.2, 0.8, 0.8]),
    CellTypes.color_b | CellTypes.color_r: np.array([0.8, 0.2, 0.8]),
    CellTypes.rainbow_color: np.array([1.0, 1.0, 1.0])
}

background_colors = {
    CellTypes.empty: np.array([0.6, 0.6, 0.6]),
    CellTypes.color_r: np.array([0.9, 0.6, 0.6]),
    CellTypes.color_g: np.array([0.6, 0.9, 0.6]),
    CellTypes.color_b: np.array([0.5, 0.5, 0.9]),
    CellTypes.color_r | CellTypes.color_g: np.array([0.9, 0.9, 0.6]),
    CellTypes.color_g | CellTypes.color_b: np.array([0.6, 0.9, 0.9]),
    CellTypes.color_b | CellTypes.color_r: np.array([0.9, 0.6, 0.9]),
    CellTypes.rainbow_color: np.array([0.9, 0.9, 0.9])
}


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


def render_board(board, goals=0, orientation=0):
    """
    Just render the board itself. Doesn't require game state.
    """
    data = render_cell(board, goals, orientation)
    data = np.hstack(tuple(data))
    data = np.hstack(tuple(data))
    return data


def render_game(game):
    return render_board(game.board, game.goals, game.orientation)
