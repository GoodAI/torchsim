import numpy as np
from typing import List

import torch
from matplotlib import pyplot as plt

from torchsim.core.eval.series_plotter import plot_multiple_runs
from torchsim.gui.observer_system import MatplotObservable


class LinePlotObserver(MatplotObservable):
    """Plots each row of a 2D tensor as a line into one plot."""
    _matrix: torch.Tensor

    def __init__(self, matrix: torch.Tensor):
        self._matrix = matrix

    def get_data(self) -> plt:
        plt.close('all')

        data = self._matrix.cpu().numpy()
        n_steps = data.shape[1]

        x_vlaues = np.arange(0, n_steps)
        figure = plot_multiple_runs(x_vlaues, data)

        return figure


class BufferedLinePlotObserver(MatplotObservable):
    """Each time it is called Plots each row of a 2D tensor as a line into one plot."""
    _matrix: torch.Tensor
    _buffer: np.ndarray

    def __init__(self, data_vector: torch.Tensor):
        self._matrix = data_vector
        self._buffer = np.ndarray((data_vector.shape[0], 0))

    def get_data(self) -> plt:
        plt.close('all')

        self._buffer = np.append(self._buffer, self._matrix.cpu().numpy(), axis=1)

        n_steps = self._buffer.shape[1]

        x_vlaues = np.arange(0, n_steps)
        figure = plot_multiple_runs(x_vlaues, self._buffer)

        return figure


class StepLinePlotObserver(MatplotObservable):
    """Simple line plot observer."""
    _matrix: torch.Tensor
    _buffer: List[List[float]]

    def __init__(self):
        self._buffer = []

    def add_value(self, values: List[float]):
        """This method should be called once per step, so the x axis represents number of steps"""
        self._buffer.append(values)

    def get_data(self) -> plt:
        plt.close('all')

        # clone list to ensure consistency
        buffer = list(self._buffer)

        x_values = np.arange(0, len(buffer))
        data = np.array(buffer)
        plt.plot(x_values, data)

        return plt
