import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

from .game_physics import GameOfLife, CellTypes
from .array_utils import wrapping_array
from .gen_board import gen_game


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
    max_steps = 1200
    num_steps = 0
    goal_points = 0.1
    no_movement_penalty = 0.02
    difficulty = 2.9
    has_fences = True
    max_regions = 2
    default_channels = (0, 1, 4, 8, 10, 14)

    def __init__(self, board_size=14, view_size=15, output_channels=default_channels):
        self.output_channels = output_channels
        self.board_shape = (board_size, board_size)
        self.view_shape = (view_size, view_size)
        self.action_space = spaces.Discrete(9)
        if output_channels:
            if output_channels == "all":
                output_channels = tuple(range(15))
            self.observation_space = spaces.Box(
                low=0, high=1,
                shape=self.view_shape + (len(output_channels),),
                dtype=np.uint16,
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
        board = self.state.board.copy()
        goals = self.state.goals.copy()

        # Get rid of the frozen flag for the agent and exit.
        # (maybe a minor optimization)
        #   agent_or_exit = (board & (CellTypes.agent | CellTypes.exit)) > 0
        #   board ^= CellTypes.frozen * agent_or_exit

        # Get rid of white cells in the goals.
        # They have an effect on red life, but they make things way more
        # complicated than they need to be.
        goals *= (goals != CellTypes.rainbow_color)

        # Combine board and goals into one array
        board = board + (goals << 3)

        # And center the array on the agent.
        h, w = self.view_shape
        x0, y0 = self.state.agent_loc
        x0 -= w // 2
        y0 -= h // 2
        board = board.view(wrapping_array)[y0:y0+h, x0:x0+w]
        board = board.view(np.ndarray)

        # If the environment specifies output channels, output a boolean array
        # with the channels as the third dimension. Otherwise output a bit
        # array.
        if self.output_channels:
            shift = np.array(list(self.output_channels), dtype=np.int16)
            board = (board[...,None] & (1 << shift)) >> shift
        return board

    def step(self, action):
        assert self.state is not None, "State not initializeddef."
        old_position = self.state.agent_loc
        self.state.advance_board()
        action_name = self.action_names[action]
        base_reward = self.state.execute_action(action_name)
        new_state_value = self.state.current_points()
        base_reward += new_state_value - self.old_state_value
        self.old_state_value = new_state_value
        self.num_steps += 1
        times_up = self.num_steps >= self.max_steps
        done = self.state.game_over or times_up
        standing_still = old_position == self.state.agent_loc
        reward = base_reward - standing_still * self.no_movement_penalty
        return self._get_obs(), reward, done, {
            'did_move': not standing_still,
            'times_up': times_up,
            'base_reward': base_reward,
        }

    def reset(self):
        if self.difficulty >= 0:
            state = gen_game(
                self.board_shape, self.difficulty, self.has_fences,
                self.max_regions)
        else:
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
        # Get rid of movable blocks.
        state.board &= ~CellTypes.movable
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
