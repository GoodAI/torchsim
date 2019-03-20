import numpy
import pytest

from torchsim.core.eval.series_plotter import plot_multiple_runs, plot_multiple_runs_with_baselines
import matplotlib.pyplot as plt


def to_array_of_lists(x_vals):
    return numpy.array(x_vals)


def create_baseline(y_values):
    """Just add a small number to the y_values to create the baseline series."""

    diff = 0.25

    if type(y_values[0]) is list:
        baseline = [list(map(lambda y: y + diff, ys)) for ys in y_values]
    else:
        baseline = list(map(lambda y: y + diff, y_values))

    return baseline


def different_plots(x_values, y_values, baseline, legend):
    """Tests potentially problematic plot setups."""
    plot_multiple_runs(x_values, y_values, labels=legend)
    plot_multiple_runs(x_values, y_values, labels=legend, smoothing_window_size=3)
    plot_multiple_runs_with_baselines(x_values, y_values, baseline, labels=legend)
    plot_multiple_runs_with_baselines(x_values, y_values, baseline, labels=legend, smoothing_window_size=3)

@pytest.mark.slow
@pytest.mark.parametrize('plot_input',
                         [{'x_values': [[2, 3, 4, 5]], 'y_values': [[1, 2, 3, 4]], 'legend': ['hello']},
                          {'x_values': [2, 3, 4, 5], 'y_values': [[1, 2, 3, 4]], 'legend': ['hello']},
                          {'x_values': [2.1, 3.1, 4.1, 5.1], 'y_values': [1.1, 2.1, 3.1, 4.1], 'legend': ['hello']},
                          {'x_values': [2, 3, 4, 5], 'y_values': [[1, 2, 3, 4], [11, 12, 13, 14]], 'legend': ['a', 'b']},
                          {'x_values': [[2, 3, 4, 5], [3, 4, 5, 6, 7]], 'y_values': [[1, 2, 3, 4], [11, 12, 13, 14, 15]],
                           'legend': ['hello', 'hello2']}]
                         )
def test_plot_multiple_runs(plot_input):
    """Tests various combinations of inputs to the series plotter."""

    show_figures = False

    x_values = plot_input['x_values']
    y_values = plot_input['y_values']
    legend = plot_input['legend']

    # plot list (of lists)
    base = create_baseline(y_values)
    different_plots(x_values, y_values, create_baseline(y_values), legend)

    # plot np.ndarray of lists
    x_vals, y_vals = to_array_of_lists(x_values), to_array_of_lists(y_values)
    base_np = to_array_of_lists(base)
    different_plots(x_vals, y_vals, base_np, legend)

    if show_figures:
        plt.show()
    else:
        plt.close('all')


