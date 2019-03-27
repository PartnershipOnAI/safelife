import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

from .game_physics import GameOfLife, CellTypes
from .array_utils import wrapping_array


class GameOfLifeEnv(gym.Env):
    """
    ...TK
    """

    metadata = {
        'render.modes': ['ansi', 'rgb_array'],
        'video.frames_per_second': 30
    }
    action_names = [
        "NULL",
        "MOVE UP",
        "MOVE RIGHT",
        "MOVE DOWN",
        "MOVE LEFT",
        "TOGGLE UP",
        "TOGGLE RIGHT",
        "TOGGLE DOWN",
        "TOGGLE LEFT",
    ]
    state = None
    old_state_value = 0
    max_steps = 3000
    num_steps = 0
    goal_points = 0.05  # deemphasize level exit points

    def __init__(self, board_size=12, view_size=15, output_channels=(0,1,14)):
        # For now, default to only 3 channels: life, agent, and goal.
        self.output_channels = output_channels
        self.board_shape = (board_size, board_size)
        self.view_shape = (view_size, view_size)
        self.action_space = spaces.Discrete(9)
        if output_channels:
            if output_channels == "all":
                output_channels = tuple(range(15))
            self.observation_space = spaces.Box(
                low=0, high=2**15,
                shape=self.view_shape + (len(output_channels),),
                dtype=np.uint8,
            )
        else:
            self.observation_space = spaces.Box(
                low=0, high=2**15,
                shape=self.view_shape,
                dtype=np.uint16,
            )
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_obs(self):
        board = self.state.board + (self.state.goals << 3)
        h, w = self.view_shape
        x0, y0 = self.state.agent_loc
        x0 -= w // 2
        y0 -= h // 2
        board = board.view(wrapping_array)[y0:y0+h, x0:x0+w]
        board = board.view(np.ndarray)
        if self.output_channels:
            shift = np.array(list(self.output_channels), dtype=np.int16)
            board = (board[...,None] & (1 << shift)) >> shift
        return board

    def step(self, action):
        assert self.state is not None, "State not initializeddef."
        self.state.advance_board()
        action_name = self.action_names[action]
        reward = self.state.execute_action(action_name)
        new_state_value = self.state.current_points()
        reward += new_state_value - self.old_state_value
        self.old_state_value = new_state_value
        self.num_steps += 1
        done = self.state.game_over or self.num_steps >= self.max_steps
        return self._get_obs(), reward, done, {}

    def reset(self):
        state = GameOfLife(self.board_shape)
        # For now, just add in a random 2x2 block and an exit.
        # Note that they might be overlapping
        i0 = np.random.randint(1, self.board_shape[0]-1)
        j0 = np.random.randint(1, self.board_shape[1]-1)
        state.goals[i0:i0+2, j0:j0+2] = CellTypes.color_b
        i1 = np.random.randint(1, self.board_shape[0])
        j1 = np.random.randint(1, self.board_shape[1])
        state.board[i1,j1] = CellTypes.level_exit
        state.points_on_level_exit = self.goal_points
        self.state = state
        self.old_state_value = state.current_points()
        self.num_steps = 0
        return self._get_obs()

    def render(self, mode='ansi'):
        if mode == 'ansi':
            from .asci_renderer import render_board
            return render_board(self.state, view_size=self.view_shape)
        else:
            from .rgb_renderer import render_game
            return render_game(self.state)

    def close(self):
        pass


def test_run(logdir=None):
    """
    A quick test to show that the environment is working properly
    """
    import shutil
    import os
    import gym.wrappers

    if logdir is None:
        logdir = os.path.abspath(os.path.join(__file__, '../../data/gym-test/'))
    if os.path.exists(logdir):
        shutil.rmtree(logdir)
    env = gym.wrappers.Monitor(GameOfLifeEnv(), logdir)
    env.reset()
    done = False
    while not done:
        obs, reward, done, info = env.step(env.action_space.sample())
    env.close()


if __name__ == "__main__":
    test_run()
