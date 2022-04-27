"""A `dowel.logger.LogOutput` for wandb.
It receives the input data stream from `dowel.logger`, then logs it to the Weights and Biases online dashboard
"""
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
            self._waiting_for_dump.append(data)
        else:
            raise ValueError('Unacceptable type.')

    def _record_tabular(self, data_lst):
        all_data = {}
        total_size = 0
        for data in data_lst:
            all_data.update(data.as_dict)
            total_size += len(data.as_dict)
            for key, value in data.as_dict.items():
                data.mark(key)

        # check that the keys are all disjoint
        assert len(all_data) == total_size
        wandb.log(all_data)


    def dump(self, step=None):
        """Flush summary writer to disk."""
        # Log the tabular inputs, now that we have a step
        self._record_tabular(self._waiting_for_dump)
        self._waiting_for_dump.clear()
        self._default_step += 1
