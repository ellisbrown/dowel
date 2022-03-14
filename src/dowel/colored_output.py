"""A colorized version of the `dowel.logger.StdOutput`."""

import datetime
from inspect import getframeinfo, stack

import colorama
import dateutil
import termcolor

from dowel.simple_outputs import StdOutput
from dowel.tabular_input import TabularInput

colorama.init(autoreset=True)

# options (see termcolor.colored()): `color`, `on_color`, `attrs`
COLOR_DEFAULTS = {
    'timestamp': {
        'color': 'green'
    },
    'prefix': {
        'color': 'grey',
        'attrs': ['bold']
    },
    'tabular': {
        'attrs': ['bold']
    },
    'text': {
        'color': 'white'
    },
    'callstack': {
        'color': 'blue'
    },
}

CALLSTACK_DEFAULTS = {'with_function_name': True, 'depth': 3}


def stack_info(depth: int = 1, with_function_name: bool = True) -> str:
    """Get the stack info."""
    assert depth >= 0, 'callstack_depth must be >= 0'
    stack_info = getframeinfo(stack()[depth][0])
    filename = stack_info.filename.split('/')[-1]
    infostr = f'{filename}#L{stack_info.lineno}'
    if with_function_name:
        infostr += f'.{stack_info.function}()'
    return infostr


class ColoredStdOutput(StdOutput):
    """Colorized console output for the logger.

    :param with_timestamp: Whether to log a timestamp before non-tabular data.
    :param with_callstack: Whether to log the callstack before non-tabular
        data.
    :param with_function_name: Whether to log the function name at the end of
        the callstack. If `with_callstack` is if `False`, this parameter is
        ignored.
    :param color_overrides: A dictionary of color overrides. Defaults to
        `COLOR_DEFAULTS`.
    """

    def __init__(self,
                 with_timestamp=True,
                 with_callstack=True,
                 callstack_overrides={},
                 color_overrides={}):
        self._with_timestamp = with_timestamp
        self._with_callstack = with_callstack
        self._callstack_config = {**CALLSTACK_DEFAULTS, **callstack_overrides}
        self._color_config = {**COLOR_DEFAULTS, **color_overrides}

    def colorize(self, text, key):
        """Colorize text according to the config."""
        if key in self._color_config:
            return termcolor.colored(text, **self._color_config[key])
        else:
            return text

    def record(self, data, prefix=''):
        """Log data to console."""
        if isinstance(data, str):
            pre = ''
            if self._with_timestamp:
                now = datetime.datetime.now(dateutil.tz.tzlocal())
                timestamp = now.strftime('%Y-%m-%d %H:%M:%S')
                pre = self.colorize(timestamp, 'timestamp') + ' '
            if self._with_callstack:
                callstack = stack_info(
                    self._callstack_config['depth'],
                    self._callstack_config['with_function_name'])
                callstack = self.colorize(callstack, 'callstack')
                pre += callstack + ' '
            if prefix != '':
                prefix = self.colorize(prefix, 'prefix')
                pre += prefix + ' '
            out = pre + self.colorize(data, 'text')
        elif isinstance(data, TabularInput):
            out = self.colorize(data, 'tabular')
            data.mark_str()
        else:
            raise ValueError('Unacceptable type')

        print(out)
