"""A `dowel.logger.LogOutput` for wandb.
It receives the input data stream from `dowel.logger`, then logs it to the Weights and Biases online dashboard
"""
import functools
import numpy as np
import wandb
from dowel.logger import LoggerWarning, LogOutput
from dowel.tabular_input import TabularInput


class WandbOutput(LogOutput):
    """Wandb output for logger."""

    def __init__(self):
        super().__init__()
        self._waiting_for_dump = []
        self._default_step = 0

    @property
    def types_accepted(self):
        """Return the types that the logger may pass to this output."""
        return (TabularInput,)

    def record(self, data, prefix=''):
        """Add data to wandb summary.
        Args:
            data: The data to be logged by the output.
            prefix(str): Not used.
        """
        if isinstance(data, TabularInput):
            self._waiting_for_dump.append(
                functools.partial(self._record_tabular, data))
        else:
            raise ValueError('Unacceptable type.')

    def _record_tabular(self, data, step):
        for key, value in data.as_dict.items():
            self._record_kv(key, value, step)
            data.mark(key)

    def _record_kv(self, key, value, step):
        if isinstance(value, np.ScalarType):
            wandb.log({key: value}, step=step)
        else:
            raise NotImplementedError

    def dump(self, step=None):
        """Flush summary writer to disk."""
        # Log the tabular inputs, now that we have a step
        for p in self._waiting_for_dump:
            p(step or self._default_step)
        self._waiting_for_dump.clear()
        self._default_step += 1
