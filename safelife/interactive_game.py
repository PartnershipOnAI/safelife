import os
import sys
import glob
import textwrap
import time
import re
from types import SimpleNamespace
from collections import defaultdict, deque
import numpy as np

from .safelife_game import SafeLifeGame, ORIENTATION
from . import render_text
from . import render_graphics
from .keyboard_input import KEYS, getch
from .side_effects import side_effect_score
from .level_iterator import SafeLifeLevelIterator
from .random import set_rng
from .safelife_logger import StreamingJSONWriter, combined_score


COMMAND_KEYS = {
    KEYS.LEFT_ARROW: "LEFT",
    KEYS.RIGHT_ARROW: "RIGHT",
    KEYS.UP_ARROW: "UP",
    KEYS.DOWN_ARROW: "DOWN",
    '\r': "NULL",
    ' ': "NULL",
    'c': "TOGGLE",
    'R': "RESTART",
    '>': "NEXT LEVEL",
    '<': "PREV LEVEL",
}

EDIT_KEYS = {
    KEYS.LEFT_ARROW: "MOVE LEFT",
    KEYS.RIGHT_ARROW: "MOVE RIGHT",
    KEYS.UP_ARROW: "MOVE UP",
    KEYS.DOWN_ARROW: "MOVE DOWN",
    'x': "PUT EMPTY",
    'a': "PUT AGENT",
    'c': "PUT LIFE",
    'C': "PUT HARD LIFE",
    'w': "PUT WALL",
    'r': "PUT CRATE",
    'e': "PUT EXIT",
    'i': "PUT ICECUBE",
    't': "PUT PLANT",
    'T': "PUT TREE",
    'd': "PUT WEED",
    'p': "PUT PARASITE",
    'f': "PUT FOUNTAIN",
    'n': "PUT SPAWNER",
    'N': "PUT HARD SPAWNER",
    '1': "TOGGLE AGENT",
    '2': "TOGGLE ALIVE",
    '3': "TOGGLE PUSHABLE",
    '4': "TOGGLE PULLABLE",
    '5': "TOGGLE DESTRUCTIBLE",
    '6': "TOGGLE FROZEN",
    '7': "TOGGLE PRESERVING",
    '8': "TOGGLE INHIBITING",
    '9': "TOGGLE SPAWNING",
    '0': "TOGGLE EXIT",
    '[': "PREVIOUS EDIT COLOR",
    ']': "NEXT EDIT COLOR",
    ';': "APPLY EDIT COLOR",
    's': "SAVE",
    'S': "SAVE AS",
    'R': "REVERT",
    'Q': "ABORT LEVEL",
    '>': "NEXT LEVEL",
    '<': "PREV LEVEL",
}

TOGGLE_EDIT = ('~', '`')
SAVE_RECORDING = '*'
START_SHELL = '\\'
HELP_KEYS = ('?', '/')
UNDO_KEY = 'z'

MAX_HISTORY_LENGTH = 10000


class GameLoop(object):
    """
    Play the game interactively. For humans.
    """
    load_from = None
    view_size = None
    centered_view = False
    gen_params = None
    print_only = False
    relative_controls = True
    can_edit = True
    recording_directory = "plays"  # in the current working directory

    side_effect_weights = {
        'life-green': 1.0,
        'spawner-yellow': 2.0,
    }
    use_wandb = False

    def __init__(self, level_generator):
        self.level_generator = level_generator
        self.loaded_levels = []
        self.state = SimpleNamespace(
            screen="INTRO",
            game=None,
            total_points=0,
            total_steps=0,
            total_safety_score=0,
            level_start_steps=0,
            level_start_points=0,
            total_undos=0,
            edit_mode=0,
            history=deque(maxlen=MAX_HISTORY_LENGTH),
            side_effects=None,
            total_side_effects=defaultdict(lambda: np.zeros(2)),
            message="",
            last_command="",
            level_num=0,
            episode_stats=[],
        )

    def load_next_level(self, incr=1):
        self.state.level_num = max(1, self.state.level_num + incr)
        if self.state.level_num <= len(self.loaded_levels):
            self.state.game = self.loaded_levels[self.state.level_num-1]
            self.state.game.revert()
        else:
            self.state.game = next(self.level_generator)
            self.loaded_levels.append(self.state.game)
        if len(self.state.game.agent_locs) > 0:
            self.state.game.edit_loc = tuple(self.state.game.agent_locs[0])
        self.state.level_start_points = self.state.total_points
        self.state.level_start_steps = self.state.total_steps
        self.state.level_start_undos = self.state.total_undos
        self.state.level_start_time = time.time()
        self.state.history.clear()
        self.record_frame()

    def next_recording_name(self):
        pattern = os.path.join(self.recording_directory, 'rec-*.npz')
        old_recordings = glob.glob(pattern)
        if not old_recordings:
            n = 1
        else:
            n = max(
                int(os.path.split(fname)[1][4:-4])
                for fname in old_recordings
            ) + 1
        fname = 'rec-{:03d}.npz'.format(n)
        return os.path.join(self.recording_directory, fname)

    def record_frame(self, restart=False):
        state = self.state
        game = state.game
        if game is None:
            return
        snapshot = game.serialize()
        snapshot['num_steps'] = game.num_steps
        snapshot['total_steps'] = state.total_steps
        snapshot['total_points'] = state.total_points
        snapshot['is_restart'] = restart
        state.history.append(snapshot)

    def save_recording(self):
        boards = []
        goals = []
        agent_locs = []
        for snapshot in reversed(self.state.history):
            boards.append(snapshot['board'])
            goals.append(snapshot['goals'])
            agent_locs.append(snapshot['agent_locs'])
            if snapshot['is_restart']:
                break
        if not boards:
            return
        data = {
            'board': boards[::-1],
            'goals': goals[::-1],
            'agent_locs': agent_locs[::-1],
        }

        pattern = os.path.join(self.recording_directory, 'rec-*.npz')
        old_recordings = glob.glob(pattern)
        if not old_recordings:
            n = 1
        else:
            n = max(
                int(os.path.split(fname)[1][4:-4])
                for fname in old_recordings
            ) + 1
        fname = 'rec-{:03d}.npz'.format(n)
        next_recording_name = os.path.join(self.recording_directory, fname)

        os.makedirs(self.recording_directory, exist_ok=True)
        np.savez_compressed(next_recording_name, **data)
        return next_recording_name

    def log_level_stats(self):
        state = self.state
        game = state.game
        if not game:
            return

        reward = np.average(game.points_earned())
        reward_possible = np.average(
            game.initial_available_points() + game.points_on_level_exit)

        long_level_stats = {
            'level_name': game.title,
            'length': state.total_steps - state.level_start_steps,
            'undos': state.total_undos - state.level_start_undos,
            'reward': reward,
            'reward_possible': reward_possible,
            'reward_needed': np.average(game.required_points()),
            'delta_time': time.time() - state.level_start_time,
            'side_effects': state.side_effects,
        }

        short_level_stats = {
            'length': state.total_steps - state.level_start_steps,
            'undos': state.total_undos - state.level_start_undos,
            'reward': reward / reward_possible,
        }
        # The "score" here is the combined safety/performance score.
        # Matches the combined score for trained agents.
        short_level_stats['side_effects'], short_level_stats['score'] = combined_score(
            long_level_stats, self.side_effect_weights)
        state.episode_stats.append(short_level_stats)

        if self.logfile:
            writer = StreamingJSONWriter(self.logfile)
            writer.dump(long_level_stats)
            writer.close

        if self.use_wandb:
            import wandb
            data = {
                'human/'+key: val for key, val in short_level_stats.items()
            }
            recording = self.save_recording()
            render_graphics.render_file(recording, fps=15, movie_format="mp4")
            data['human/video'] = wandb.Video(recording[:-4] + '.mp4')
            wandb.log(data)

    def log_final_level_stats(self):
        stats = self.state.episode_stats
        if stats and self.use_wandb:
            import wandb
            wandb.run.summary['avg_length'] = np.average(
                [x['length'] for x in stats])
            wandb.run.summary['reward'] = np.average(
                [x['reward'] for x in stats])
            wandb.run.summary['side_effects'] = np.average(
                [x['side_effects'] for x in stats])
            wandb.run.summary['score'] = np.average(
                [x['score'] for x in stats])
            wandb.run.summary['undos'] = np.sum(
                [x['undos'] for x in stats])
            wandb.run.finish()

    def undo(self):
        history = self.state.history
        game = self.state.game
        if len(history) < 2 or game is None:
            return False
        history.pop()
        snapshot = history[-1]
        game.deserialize(snapshot, as_initial_state=False)
        game.num_steps = snapshot['num_steps']
        self.state.total_points = snapshot['total_points']
        self.state.total_steps = snapshot['total_steps']
        self.state.total_undos += 1
        return True

    def handle_input(self, key):
        state = self.state
        state.message = ""
        state.last_command = ""
        is_repeatable_key = False
        if key == KEYS.INTERRUPT:
            exit()
        elif self.print_only:
            # Hit any key to get to the next level
            try:
                self.load_next_level()
                state.screen = "GAME"
            except StopIteration:
                exit()
        elif key in HELP_KEYS:
            # Switch to the help screen. Will later pop the state.
            if state.screen != "HELP":
                state.prior_screen = state.screen
                state.screen = "HELP"
        elif state.screen in ("INTRO", "LEVEL SUMMARY"):
            # Hit any key to get to the next level
            try:
                self.load_next_level()
                state.screen = "GAME"
            except StopIteration:
                state.game = None
                state.screen = "GAMEOVER"
                self.log_final_level_stats()
        elif state.screen == "HELP":
            # Hit any key to get back to prior state
            state.screen = state.prior_screen
        elif key == SAVE_RECORDING:
            rec_name = self.save_recording()
            if rec_name:
                state.message = "Recording saved: " + rec_name
            else:
                state.message = "Nothing to record."
        elif key in TOGGLE_EDIT:
            if not self.can_edit:
                state.message = "edit mode disabled"
            elif not state.edit_mode:
                state.edit_mode = "BOARD"
                if state.game and len(state.game.agent_locs) > 0:
                    state.game.edit_loc = tuple(state.game.agent_locs[0])
            elif state.edit_mode == "BOARD":
                state.edit_mode = "GOALS"
            else:
                state.edit_mode = None
        elif key == START_SHELL and state.edit_mode:
            # Handle the shell command later on in the event loop
            # after we've had a chance to display something on the screen.
            state.last_command = "SHELL"
        elif state.screen == "CONFIRM_SAVE":
            if key in ('y', 'Y'):
                state.game.save(state.game.file_name)
                state.message = "Saved successfully."
            else:
                state.message = "Save aborted."
            state.screen = "GAME"
        elif key == UNDO_KEY and state.screen == "GAME":
            state.last_command = "UNDO"
            is_repeatable_key = True
            self.undo()
        elif state.screen == "GAME":
            game = state.game
            if state.edit_mode and key in EDIT_KEYS:
                # Execute action immediately.
                command = EDIT_KEYS[key]
                state.last_command = command
                if state.edit_mode == "GOALS":
                    command = "GOALS " + command
                if command.startswith("SAVE"):
                    if command == "SAVE" and (
                            game.file_name and game.file_name.endswith('.npz')):
                        state.screen = "CONFIRM_SAVE"
                        short_name = os.path.split(game.file_name)[1]
                        state.message = "Save level as '%s'? (y/n)" % short_name
                    else:
                        # User will have to go to the terminal, but oh well.
                        # Not worth the effort to manage text input.
                        state.last_command = "SAVE AS"
                else:
                    state.message = game.execute_edit(command) or ""
                if command.startswith("MOVE"):
                    is_repeatable_key = True
                else:
                    self.record_frame()
            elif COMMAND_KEYS.get(key) in ("NEXT LEVEL", "PREV LEVEL"):
                command = COMMAND_KEYS[key]
                state.last_command = command
                if not self.can_edit:
                    state.message = "Level skipping is disabled."
                else:
                    state.total_points += game.execute_action(command)
            elif not state.edit_mode and key in COMMAND_KEYS:
                command = COMMAND_KEYS[key]
                needs_board_advance = True
                if command in ("LEFT", "RIGHT", "UP", "DOWN"):
                    command_orientation = ORIENTATION[command]
                    if self.relative_controls and command in ("LEFT", "RIGHT"):
                        needs_board_advance = False
                        command = "TURN " + command
                    elif self.relative_controls:
                        command = {
                            "UP": "MOVE FORWARD",
                            "DOWN": "MOVE BACKWARD",
                        }[command]
                    elif command_orientation != game.orientation:
                        needs_board_advance = False
                        command = "FACE " + command
                    else:
                        command = "MOVE " + command
                state.last_command = command
                if needs_board_advance:
                    state.total_steps += 1
                    start_pts = np.sum(game.current_points())
                    action_pts = game.execute_action(command)
                    game.advance_board()
                    end_pts = np.sum(game.current_points())
                    state.total_points += (end_pts - start_pts) + action_pts
                    self.record_frame()
                else:
                    state.total_points += game.execute_action(command)
            is_repeatable_key = not game.game_over
            if game.game_over == "RESTART":
                game.revert()
                state.total_points = state.level_start_points
                state.total_steps = state.level_start_steps
                self.record_frame(restart=True)
            elif game.game_over in ("ABORT LEVEL", "NEXT LEVEL"):
                try:
                    self.load_next_level()
                except StopIteration:
                    state.game = None
                    state.screen = "GAMEOVER"
                    self.log_final_level_stats()
            elif game.game_over == "PREV LEVEL":
                self.load_next_level(-1)
            elif game.game_over:
                state.screen = "LEVEL SUMMARY"
                state.side_effects = side_effect_score(game, strkeys=True)
                for key, val in state.side_effects.items():
                    state.total_side_effects[key] += val
                self.log_level_stats()

        if not is_repeatable_key:
            self.last_key_down = self.last_key_modifier = None

    def handle_save_as(self):
        state = self.state
        game = state.game
        state.last_command = ""
        if game is None:
            return
        print("\nCurrently file is: ", game.file_name)
        save_name = input("Save as: ")
        if save_name:
            try:
                game.save(save_name)
                state.message = "Saved successfully."
            except FileNotFoundError as err:
                state.message = "No such file or directory: '%s'" % (err.filename,)
        else:
            state.message = "Save aborted."
        print(state.message)
        self.set_needs_display()

    def handle_shell(self):
        state = self.state
        game = state.game  # noqa, just tee up local variables for the shell
        state.last_command = ""
        self.set_needs_display()
        try:
            from IPython import embed
        except ImportError:
            print("Could not import IPython.")
            print("Hit any key to return to the game.")
            getch()
        else:
            embed()

    intro_text = """
    ##########################################################
    ##                       SafeLife                       ##
    ##########################################################

    Use the arrow keys to move, 'c' to create or destroy life,
    and 'enter' to stand still. Try to add cells to blue goals
    and remove unwanted red cells, and try not to make too big
    of a mess!

    (Hit '?' to access help, 'esc' to quit, or any other key
    to continue.)
    """

    help_text = """
    Play mode
    ---------
    arrows:  movement            c:  create / destroy
    return:  wait                R:  restart level

    z:  undo
    ~:  toggle edit mode
    *:  save recording

    Edit mode
    ---------
    x:  clear cell               0-9:  toggle cell properties
    a:  add agent                []:  change edit color
    c:  add life                 ;:  apply edit color
    C:  add hard life            v:  apply edit color
    w:  add wall
    r:  add crate                s:  save
    e:  add exit                 S:  save as (in terminal)
    i:  add icecube              R:  revert level
    t:  add plant                >:  skip to next level
    T:  add tree                 <:  back to previous level
    p:  add parasite
    f:  add fountain             \\:  enter shell
    n:  add spawner

                (hit any key to continue)
    """

    @property
    def effective_view_size(self):
        if self.state.game is None:
            return None
        elif self.view_size:
            return self.view_size
        elif self.centered_view:
            return self.state.game.board.shape
        else:
            return None

    def above_game_message(self, styled=True):
        state = self.state
        game = state.game
        styles = {
            'bold': '\x1b[1m',
            'italics': '\x1b[3m',
            'clear': '\x1b[0m',
        } if styled else {
            'bold': '',
            'italics': '',
            'clear': '',
        }
        if game is None:
            return " "
        elif game.title:
            output = "{bold}{}{clear}".format(game.title, **styles)
        else:
            output = "{bold}Board #{}{clear}".format(state.level_num, **styles)
        if self.print_only:
            output += "\n"
        else:
            available_points = np.average(game.initial_available_points())
            earned_points = np.average(game.points_earned())
            pfrac = earned_points / available_points if available_points > 0 else 1.0
            output += (
                "\n  Steps: {bold}{}{clear}"
                "    Points: {bold}{}{clear}"
                "    Completed: {bold}{:0.0%}{clear}"
            ).format(game.num_steps, earned_points, pfrac, **styles)
            if state.edit_mode:
                output += "\n\n{bold}*** EDIT {} ***{clear}".format(state.edit_mode, **styles)
                output += "\n  {italics}{}{clear}".format(
                    render_text.edit_details(game, state.edit_mode), **styles)
        return output

    def below_game_message(self):
        if self.state.message:
            return self.state.message + '\n'
        elif self.state.last_command:
            return 'Action: ' + self.state.last_command + '\n'
        else:
            return '\n'

    def print_side_effects(self, side_effects, ansi=True):
        output = ""
        fmt = "    {name:14s} {val:6.2f}\n"
        for name, score in side_effects.items():
            score = score[0]
            if not ansi:
                output += fmt.format(name=name+':', val=score)
            else:
                # Formatted padding doesn't really work since we use
                # extra escape characters. Use replace instead.
                line = fmt.format(name='zz:', val=score)
                output += line.replace('zz', str(name))
        return output

    def gameover_message(self, ansi=True):
        state = self.state
        output = textwrap.dedent("""
            Game over!
            ----------

            Total points: {}
            Total steps: {}
            Total undos: {}
            Total side effects:

            {}
        """)[1:-1].format(
            state.total_points, state.total_steps, state.total_undos,
            self.print_side_effects(state.total_side_effects, ansi),
        )
        return output

    def level_summary_message(self, ansi=True):
        state = self.state
        game = state.game
        if game is None:
            return ""

        output = ""
        if game.title:
            output = "Level: %s\n\n" % (game.title,)
        output += textwrap.dedent("""
            Points: {} / {}
            Steps: {}
            Undos: {}

            Side effect scores (lower is better):

            {}

            (hit any key to continue)
        """)[1:-1].format(
            np.average(game.points_earned()),
            np.average(game.initial_available_points() + game.points_on_level_exit),
            game.num_steps,
            state.total_undos - state.level_start_undos,
            self.print_side_effects(state.side_effects, ansi)
        )
        return output

    def render_text(self):
        if not self.print_only:
            output = "\x1b[H\x1b[J"
        else:
            output = "\n"
        state = self.state
        if state.screen == "INTRO":
            output += self.intro_text
        elif state.screen == "HELP":
            output += self.help_text
        elif state.screen in ("GAME", "CONFIRM_SAVE") and state.game is not None:
            game = state.game
            game.update_exit_colors()
            output += self.above_game_message(styled=True) + '\n'
            output += render_text.render_game(
                state.game, self.effective_view_size, state.edit_mode)
            output += "\n"
            if not self.print_only:
                output += self.below_game_message()
        elif state.screen == "LEVEL SUMMARY" and state.side_effects is not None:
            output += self.level_summary_message()
        elif state.screen == "GAMEOVER":
            output += '\n\n' + self.gameover_message()
        sys.stdout.write(output)
        sys.stdout.flush()

    def setup_run(self):
        if self.print_only:
            try:
                self.load_next_level()
                self.state.screen = "GAME"
            except StopIteration:
                self.state.screen = None
                print("No game boards to print")

    def run_text(self):
        self.setup_run()
        if not self.print_only:
            os.system('clear')
        self.render_text()
        while self.state.screen != "GAMEOVER":
            self.handle_input(getch())
            if self.state.last_command == "SHELL":
                self.handle_shell()
            elif self.state.last_command == "SAVE AS":
                self.handle_save_as()
            self.render_text()

    def render_gl(self):
        # Note that this routine is pretty inefficient. It should be fine
        # for moderately large (25x25) boards, but it'll get sluggish for
        # anything really big.
        import pyglet
        import pyglet.gl as gl
        state = self.state
        window = self.window
        min_width = 550  # not a brilliant way to handle text, but oh well.

        if self.needs_display < 1:
            return
        self.needs_display -= 1

        def fullscreen_msg(msg):
            pyglet.text.Label(msg,
                font_name='Courier', font_size=11,
                x=window.width//2, y=window.height//2,
                width=min(window.width*0.9, min_width),
                anchor_x='center', anchor_y='center', multiline=True).draw()

        def render_img(img, x, y, w, h):
            img_data = pyglet.image.ImageData(
                img.shape[1], img.shape[0], 'RGB', img.tobytes())
            tex = img_data.get_texture()
            gl.glEnable(gl.GL_TEXTURE_2D)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
            gl.glBindTexture(gl.GL_TEXTURE_2D, tex.id)
            pyglet.graphics.draw_indexed(4, gl.GL_TRIANGLE_STRIP,
                [0, 1, 2, 0, 2, 3],
                ('v2f', (x, y+h, x+w, y+h, x+w, y, x, y)),
                ('t3f', tex.tex_coords),
            )
            gl.glDisable(gl.GL_TEXTURE_2D)

        def overlay_text(line1, line2):
            l1 = pyglet.text.Label(line1,
                font_name='Courier', font_size=18,
                x=window.width / 2, y=window.height / 2,
                anchor_x='center', anchor_y='bottom')
            l2 = pyglet.text.Label(line2,
                font_name='Courier', font_size=11,
                x=window.width / 2, y=window.height / 2 - 4,
                anchor_x='center', anchor_y='top')
            w = max(l1.content_width, l2.content_width) + 15
            x1 = (window.width - w) / 2
            y1 = window.height / 2 - 4 - l2.content_height - 5
            y2 = window.height / 2 + l1.content_height + 5
            render_img(np.zeros((2,2,3), dtype=np.uint8), x1, y1, w, y2-y1)
            l1.draw()
            l2.draw()

        window.clear()
        if state.screen == "INTRO":
            fullscreen_msg(textwrap.dedent(self.intro_text[1:-1]))
        elif state.screen == "HELP":
            fullscreen_msg(textwrap.dedent(self.help_text[1:-1]))
        elif state.screen == "LEVEL SUMMARY" and state.side_effects is not None:
            fullscreen_msg(self.level_summary_message(ansi=False))
        elif state.screen == "GAMEOVER":
            fullscreen_msg(self.gameover_message(ansi=False))
        elif state.screen in ("GAME", "CONFIRM_SAVE") and state.game is not None:
            top_label = pyglet.text.Label(self.above_game_message(styled=False),
                font_name='Courier', font_size=11,
                x=window.width*0.05, y=window.height-5, width=window.width*0.9,
                anchor_x='left', anchor_y='top', multiline=True)
            top_label.draw()
            bottom_label = pyglet.text.Label(self.below_game_message(),
                font_name='Courier', font_size=11,
                x=window.width*0.05, y=5, width=window.width*0.9,
                anchor_x='left', anchor_y='bottom', multiline=True)
            bottom_label.draw()

            state.game.update_exit_colors()
            img = render_graphics.render_game(
                state.game, self.effective_view_size, state.edit_mode)
            margin_top = 10 + top_label.content_height
            margin_bottom = 10 + bottom_label.content_height
            x0 = 0
            w = window.width
            h = window.height - margin_top - margin_bottom
            if h / img.shape[0] > w / img.shape[1]:
                # constrain to width
                h = w * img.shape[0] / img.shape[1]
            else:
                w = h * img.shape[1] / img.shape[0]
            x0 = (window.width - w) / 2
            y0 = window.height - h - margin_top
            render_img(img, x0, y0, w, h)

        if state.last_command == "SAVE AS":
            overlay_text("SAVE AS...", "(go to terminal)")
        elif state.last_command == "SHELL":
            overlay_text("START SHELL", "(go to terminal)")

    def pyglet_key_down(self, symbol, modifier, repeat_in=0.3):
        from pyglet.window import key
        self.last_key_down = symbol
        self.last_key_modifier = modifier
        self.next_key_repeat = time.time() + repeat_in
        is_ascii = 27 <= symbol < 255
        char = {
            key.LEFT: KEYS.LEFT_ARROW,
            key.RIGHT: KEYS.RIGHT_ARROW,
            key.UP: KEYS.UP_ARROW,
            key.DOWN: KEYS.DOWN_ARROW,
            key.ENTER: '\r',
            key.RETURN: '\r',
            key.BACKSPACE: chr(127),
        }.get(symbol, chr(symbol) if is_ascii else None)
        if not char:
            # All other characters don't count as a key press
            # (e.g., function keys, modifier keys, etc.)
            return
        if modifier & key.MOD_SHIFT:
            char = char.upper()
        self.set_needs_display()
        self.handle_input(char)

    def handle_key_repeat(self, dt):
        from pyglet.window import key
        if self.state.last_command == "SHELL":
            return self.handle_shell()
        elif self.state.last_command == "SAVE AS":
            return self.handle_save_as()
        if time.time() < self.next_key_repeat:
            return
        symbol, modifier = self.last_key_down, self.last_key_modifier
        self.last_key_down = self.last_key_modifier = None
        if not self.keyboard[symbol]:
            return
        has_shift = self.keyboard[key.LSHIFT] or self.keyboard[key.RSHIFT]
        if bool(modifier & key.MOD_SHIFT) != has_shift:
            return
        self.pyglet_key_down(symbol, modifier, repeat_in=0.045)

    def set_needs_display(self, *args, **kw):
        # Since we're double-buffered, we need to display at least twice in
        # a row whenever there's an update. This gets decremented once in
        # each call to render_gl().
        self.needs_display = 3  # terrible hack for https://github.com/PartnershipOnAI/safelife/issues/21

    def run_gl(self):
        try:
            import pyglet
        except ImportError:
            print("Cannot import pyglet. Running text mode instead.")
            print("(hit any key to continue)")
            getch()
            self.run_text()
        else:
            self.setup_run()
            self.last_key_down = None
            self.last_key_modifier = None
            self.next_key_repeat = 0
            self.window = pyglet.window.Window(resizable=True)
            self.window.set_handler('on_draw', self.render_gl)
            self.window.set_handler('on_key_press', self.pyglet_key_down)
            self.window.set_handler('on_resize', self.set_needs_display)
            self.window.set_handler('on_show', self.set_needs_display)
            self.keyboard = pyglet.window.key.KeyStateHandler()
            self.window.push_handlers(self.keyboard)
            pyglet.clock.schedule_interval(self.handle_key_repeat, 0.02)
            pyglet.app.run()


def _make_cmd_args(subparsers):
    # used by __main__.py to define command line tools
    from argparse import RawDescriptionHelpFormatter
    long_desc = textwrap.dedent("""
    Game boards can either be specified explicitly via their file names,
    or new boards can be procedurally generated for never-ending play.
    Additionally, the view can either show the whole board a subset
    centered on the player.

    **Note**: if you wish to run SafeLife in a graphical display
    (not text-based), then you must have 'pyglet' installed.
    Use e.g. 'pip3 install pyglet'.
    """)
    desc = "Play a game of SafeLife interactively."
    play_parser = subparsers.add_parser(
        "play", help=desc, description=desc + '\n\n' + long_desc,
        formatter_class=RawDescriptionHelpFormatter)
    desc = "Generate and display new game boards."
    print_parser = subparsers.add_parser(
        "print", help=desc, description=desc + '\n\n' + long_desc,
        formatter_class=RawDescriptionHelpFormatter)
    new_parser = subparsers.add_parser(
        "new", help="Generate a new empty board of the specified size.")

    for parser in (play_parser, print_parser):
        # they use some of the same commands
        parser.add_argument('load_from',
            nargs='*', help="Load levels from file(s)."
            " Note that files can either be archived SafeLife board (.npz)"
            " or they can be parameters for procedural generation (.yaml)."
            " Files will be searched for in the 'levels'"
            " folder if not found in the current working directory."
            " Inputs of the form 'benchmark-[level]' will be treated specially,"
            " and will look for the associated benchmark file with the 'human'"
            " tag appended to it."
            " If no files are provided, a new board will be randomly generated"
            " with the default parameters.")
    for parser in (new_parser,):
        parser.add_argument('-b', '--board_size', type=int, default=15,
            help="Width and height of the empty board.",
            metavar="SIZE")
    for parser in (play_parser, new_parser):
        parser.add_argument('-a', '--absolute_controls', action='store_true',
            help="If set, use absolute instead of relative controls."
            " In relative controls, the left/right keys turn the agent and"
            " up/down move the agent forwards/backwards. In absolute controls,"
            " arrow keys either make the agent face or move in the direction"
            " indicated.")
        parser.add_argument('-c', '--centered', action='store_true',
            help="If set, the board is always centered on the agent.")
        parser.add_argument('--view_size', type=int, default=None,
            help="View size. Implies a centered view.", metavar="SIZE")
    for parser in (play_parser,):
        parser.add_argument('--logfile')
        parser.add_argument('-w', '--wandb', action="store_true",
            help="Record human play to wandb.")
    for parser in (play_parser, print_parser, new_parser):
        parser.add_argument('-t', '--text_mode', action='store_true',
            help="Run the game in the terminal instead of using a graphical"
            " display.")
        parser.add_argument('--seed', type=int, default=None,
            help="Random seed for level generation.")
        parser.set_defaults(run_cmd=_run_cmd_args)


def _run_cmd_args(args):
    seed = np.random.SeedSequence(args.seed)
    if args.cmd == "new":
        if args.board_size < 3:
            print("Error: 'board_size' must be at least 3.")
            return
        if args.board_size > 50:
            print("Error: maximum 'board_size' is 50.")
            return
        game = SafeLifeGame(board_size=(args.board_size, args.board_size))
        main_loop = GameLoop(iter([game]))
    else:
        env_type = None
        if len(args.load_from) == 1:
            # Treat inputs of the form "benchmark-levelname" specially.
            m = re.match(r'benchmark-([\w-]+)$', args.load_from[0])
            if m is not None:
                env_type = m.group(1) + '-human'
                args.load_from = [os.path.join('benchmarks', 'v1.2', env_type)]
            else:
                env_type = args.load_from[0]
        iterator = SafeLifeLevelIterator(*args.load_from, seed=seed.spawn(1)[0])
        iterator.fill_queue()
        main_loop = GameLoop(iterator)
    if args.cmd == "print":
        main_loop.print_only = True
    else:
        main_loop.centered_view = args.centered
        main_loop.relative_controls = not args.absolute_controls
        main_loop.view_size = args.view_size and (args.view_size, args.view_size)
    if args.cmd == "play":
        main_loop.logfile = args.logfile
        main_loop.use_wandb = args.wandb
        if args.wandb:
            import wandb
            wandb.init(config={'env_type': env_type, 'human_play': True})
            wandb.run.summary['env_type'] = env_type
            main_loop.recording_directory = wandb.run.dir
            main_loop.can_edit = False
    with set_rng(np.random.default_rng(seed)):
        if args.text_mode:
            main_loop.run_text()
        else:
            main_loop.run_gl()
