from itertools import permutations
from time import localtime, strftime

import matplotlib.pyplot as plt
import numpy as np
import pickle
import platform
from matplotlib.figure import Figure
import os
from typing import List, Tuple, Union, Dict, Any, Sequence, Optional

from torchsim.core.eval.doc_generator.document import Document
from torchsim.core.eval.doc_generator.figure import Image
from torchsim.utils.os_utils import project_root_dir


def get_experiment_results_folder():
    return os.path.join(project_root_dir(), 'data', 'research_topics')


def to_safe_name(filename: str):
    for c in '[]/\;,><&*:%=+@!#^(){}|?^ ':
        filename = filename.replace(c, '_')
    return filename


def to_safe_path(path_name: str) -> str:
    if 'Win' in platform.system() and '\\\\?\\' not in path_name:
        path_name = path_name.replace('/', '\\')
        path_name = "\\\\?\\" + path_name
    return path_name


def plot_multiple_runs_with_baselines(x_values: Union[List[List[float]], List[float], np.ndarray],
                                      y_values: Union[List[List[float]], List[float], np.ndarray],
                                      y_values_baseline: Union[List[List[float]], List[float], np.ndarray],
                                      xlim: List[float] = None,
                                      ylim: List[float] = None,
                                      xlabel: str = 'x',
                                      ylabel: str = 'y',
                                      title: str = '...',
                                      properties: str = '-',
                                      smoothing_window_size: int = None,
                                      labels=None,
                                      figsize: Tuple[int, int] = None,
                                      hide_legend=False,
                                      path=None,
                                      doc=None):
    """Plot baselines in light grey, then plot multiple runs on top of this and add the legend."""

    n_runs = len(to_list_list(y_values_baseline))

    fig = plot_multiple_runs(x_values=x_values,
                             y_values=y_values_baseline,
                             xlim=xlim,
                             ylim=ylim,
                             xlabel=xlabel,
                             ylabel=ylabel,
                             title=title,
                             properties=':',
                             smoothing_window_size=smoothing_window_size,
                             labels=None,
                             other_params=[{'color': (0.5, 0.5, 0.5)}] * n_runs,
                             hide_legend=True,
                             linewidth=1,
                             figsize=figsize,
                             disable_ascii_labels=True)

    fig = plot_multiple_runs(x_values=x_values,
                             y_values=y_values,
                             xlim=xlim,
                             ylim=ylim,
                             xlabel=xlabel,
                             ylabel=ylabel,
                             title=title,
                             properties=properties,
                             smoothing_window_size=smoothing_window_size,
                             labels=labels,
                             figsize=figsize,
                             hide_legend=hide_legend,
                             figure=fig,
                             disable_ascii_labels=True)

    if path is not None and doc is not None:
        add_fig_to_doc(fig, path, doc)

    return fig


def plot_with_confidence_intervals(x_values: Sequence[float],
                                   y_means: Sequence[float],
                                   y_mins: Sequence[float],
                                   y_maxes: Sequence[float],
                                   x_label: str = 'x',
                                   y_label: str = 'y',
                                   title: str = '...',
                                   path: str = None,
                                   doc=None,
                                   color='#00ccff',
                                   hide_legend: bool = False):

    fig = plt.figure()

    plt.plot(x_values, y_means, lw=1, color=color, alpha=1, label=f"{y_label} mean")
    plt.fill_between(x_values, y_mins, y_maxes, color=color, alpha=0.4, label=f"{y_label} (min, max)")

    if not hide_legend:
        plt.legend()

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.title(title)

    if path is not None and doc is not None:
        add_fig_to_doc(fig, path, doc)


def plot_multiple_runs(x_values: Union[List[Sequence[float]], List[float], np.ndarray],
                       y_values: Union[List[Sequence[float]], List[float], np.ndarray],
                       y_lower: Optional[List[Sequence[float]]] = None,
                       y_upper: Optional[List[Sequence[float]]] = None,
                       xlim: List[float] = None,
                       ylim: List[float] = None,
                       xlabel: str = 'x',
                       ylabel: str = 'y',
                       title: str = '...',
                       properties: str = '-',
                       smoothing_window_size: int = None,
                       labels=None,
                       figsize: Tuple[int, int] = None,
                       hide_legend=False,
                       linewidth=1,
                       figure=None,
                       disable_ascii_labels=False,
                       use_scatter=False,
                       other_params: List[Dict[str, Any]] = None,
                       path=None,
                       doc=None
                       ) -> Figure:
    """Plots results from multiple runs of the same experiment.

    Can plot either multiple series on y-axis against
    one series on x-axis (in that case data can be also smoothed using convolution with a uniform window) or
    multiple x-series against multiple y-series.

    Args:
        use_scatter: if true, will not show lines between points
        disable_ascii_labels:
        figure: if a figure is specified, it will add the series to it
        linewidth: width of the lines
        x_values: vector containing values on the X or matrix containing 2D array of series [n_runs, n_statistics]
        y_values: 2D array of sizes [n_runs, n_statistics]
        xlim: optional limits of the x axis
        ylim: optional limits of the y axis
        xlabel:
        ylabel:
        title:
        properties:
        smoothing_window_size: if not None, performs smoothing of each line by a square signal.
        labels:
        figsize:
        hide_legend:
        path: if both path and doc specified, the method will automatically add the figure to the given document
        doc:
    """
    x_values = to_list_list(x_values)
    y_values = to_list_list(y_values)

    draw_bounds = False

    if y_lower is not None and y_upper is not None:
        draw_bounds = True
        y_lower = to_list_list(y_lower)
        y_upper = to_list_list(y_upper)

    if smoothing_window_size is None:
        x_values_processed = x_values
        y_values_processed = y_values
        y_lower_processed = y_lower
        y_upper_processed = y_upper
    else:
        assert smoothing_window_size % 2 == 1, "Smoothing window size has to be an odd number"
        assert smoothing_window_size < len(x_values[0]), "Smoothing window has to be smaller than the signal"
        window = np.ones(smoothing_window_size) / smoothing_window_size

        y_values_processed = []
        y_lower_processed = []
        y_upper_processed = []
        for i, line in enumerate(y_values):
            y_values_processed.append(np.convolve(line, window, mode='valid'))
            if draw_bounds:
                line_lower = y_lower[i]
                line_upper = y_upper[i]
                y_lower_processed.append(np.convolve(line_lower, window, mode='valid'))
                y_upper_processed.append(np.convolve(line_upper, window, mode='valid'))

        half_size = int(smoothing_window_size / 2)
        x_values_processed = []
        for line in x_values:
            x_values_processed.append(line[half_size:-half_size])

    n_runs = len(y_values_processed)
    colors = [f'#{"".join(color_parts)}' for color_parts in permutations(['99', 'ba', '55'])] * 6

    if not hide_legend:  # in case we want to show the legend, there should be one label for one run
        assert n_runs == len(labels)

    if figure is not None:
        fig = figure
    else:
        if figsize is not None:
            fig = plt.figure(figsize=figsize)
        else:
            fig = plt.figure()

    if labels is None:
        labels = [f'run: {run}' for run in range(n_runs)]

    if other_params is None:
        other_params = [{} for _ in range(n_runs)]

    for run in range(0, n_runs):
        if len(x_values[0]) < 52 and not disable_ascii_labels:
            import string
            marker = f"${string.ascii_letters[run]}$"
        else:
            marker = ""
        if not hide_legend:
            label = f"[{run};{marker}] " + str(labels[run])
        else:
            label = None

        x_values_run = x_values_processed[0] if len(x_values_processed) == 1 else x_values_processed[run]

        if use_scatter:
            plt.scatter(x_values_run,
                        y_values_processed[run],
                        label=label,
                        **other_params[run])

            # This doesn't draw lower/upper bounds yet.
        else:
            if 'color' in other_params[run]:
                color = other_params[run]['color']
                del other_params[run]['color']
            else:
                color = colors[run]
            plt.plot(x_values_run,
                     y_values_processed[run],
                     properties,
                     color=color,
                     label=label,
                     marker=marker,
                     markersize=7,
                     linewidth=linewidth,
                     **other_params[run])

            if draw_bounds:
                plt.fill_between(x_values_run,
                                 y_lower_processed[run],
                                 y_upper_processed[run],
                                 color=colors[run],
                                 alpha=0.4)

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    if not hide_legend:
        plt.legend()

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if smoothing_window_size:
        plt.title(title + f"  | smoothing_window_size={smoothing_window_size}")
    else:
        plt.title(title)

    ax = plt.gca()
    ax.grid()

    if path is not None and doc is not None:
        add_fig_to_doc(fig, path, doc)

    return fig


def to_list_list(source: Union[List[Sequence[float]], List[float], np.ndarray]) -> List[List[float]]:
    if type(source) is np.ndarray:
        source = source.tolist()

    if type(source) is list and type(source[0]) is list:
        return source

    return [source]


def get_stamp(time_stamp: bool = True) -> str:
    """Unique time stamp for the file names."""

    stamp = ''
    if time_stamp:
        stamp = '_' + strftime("%Y-%m-%d_%H-%M-%S", localtime())
    return stamp


def save_figure(name: str, figure: None, pdf: bool = False):
    if pdf:
        figure.savefig(name + '.pdf', bbox_inches='tight')
    else:
        figure.savefig(name + '.png')


def add_fig_to_doc(fig, path_name, doc: Document):
    name = to_safe_path(path_name)
    # save figure also as pickle
    if not os.path.exists(os.path.dirname(name)):
        os.makedirs(os.path.dirname(name))
    with open(name + '.fpk', 'wb+') as file:
        pickle.dump(fig, file, protocol=2)

    name_svg = name + '.svg'
    fig.savefig(name_svg, format='svg')
    im = Image(os.path.basename(name_svg))
    doc.add(im)
