"""
Module for slightly non-standard keyboard inputs.

`Getch` is from http://code.activestate.com/recipes/134892/
"""


class KEYS:
    # keyboard constants
    UP_ARROW = '\x1b[A'
    DOWN_ARROW = '\x1b[B'
    RIGHT_ARROW = '\x1b[C'
    LEFT_ARROW = '\x1b[D'
    INTERRUPT = '\x03'
    DELETE = '\x7f'


class _Getch:
    """
    Gets a single character from standard input.  Does not echo to the screen.
    """
    def __init__(self):
        try:
            self.impl = _GetchWindows()
        except ImportError:
            self.impl = _GetchUnix()

    def __call__(self):
        c = self.impl()
        if c == '\x1b':
            c += self.impl() + self.impl()
        return c


class _GetchUnix:
    def __init__(self):
        import tty, sys  # noqa

    def __call__(self):
        import sys, tty, termios  # noqa
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


class _GetchWindows:
    def __init__(self):
        import msvcrt  # noqa

    def __call__(self):
        import msvcrt
        return msvcrt.getch()


getch = _Getch()
