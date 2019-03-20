import argparse
import glob
import os
import pickle
from os import path
from matplotlib import pyplot


def _load_and_view_figures(figure_path: str):
    print(f"Loading and visualizing figures in {path}...")
    if os.path.isdir(figure_path):
        figures_paths = list(glob.iglob(path.join(figure_path, '*.fpk')))

    else:
        figures_paths = [figure_path]

    for idx, fig_path in enumerate(figures_paths):
        print(f"Loading figure {idx} / {len(figures_paths)}")
        with open(fig_path, 'rb') as file:
            figure = pickle.load(file)
        figure.show()

    pyplot.show()

    print(f"...finished")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=None)
    args = parser.parse_args()
    _load_and_view_figures(args.path)
