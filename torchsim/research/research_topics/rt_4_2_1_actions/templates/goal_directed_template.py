from dataclasses import dataclass

import numpy as np

from itertools import islice
from os import path
from typing import List, Sequence

from torchsim.core.eval.doc_generator.document import Document
from torchsim.core.eval.series_plotter import plot_multiple_runs, plot_with_confidence_intervals
from torchsim.core.eval2.experiment_controller import ExperimentController, ExperimentComponent
from torchsim.core.eval2.experiment_template_base import ExperimentTemplateBase, TTopology
from torchsim.core.eval2.measurement_manager import MeasurementManager, RunMeasurementManager
from torchsim.research.research_topics.rt_4_2_1_actions.topologies.goal_directed_template_topology import \
    GoalDirectedTemplateTopology


class GoalDirectedTemplateMainComponent(ExperimentComponent):
    def __init__(self, topology: GoalDirectedTemplateTopology, run_measurement_manager: RunMeasurementManager,
                 window_size: int):
        self._run_measurement_manager = run_measurement_manager
        self._run_measurement_manager.add_measurement_f('rewards', topology.get_rewards)

        self._reward_counter = 0
        self._times_to_reward = []
        self._window_size = window_size

    def calculate_run_results(self):
        super().calculate_run_results()

        cumulative_reward = []
        current_accumulated_reward = 0
        rewards = self._run_measurement_manager.measurements.get_items('rewards')
        for reward in rewards:
            current_accumulated_reward += reward
            cumulative_reward.append(current_accumulated_reward)

        average_rewards = [np.array([[0]] * len(rewards[0])) for _ in range(min(self._window_size, len(rewards)))]
        np_rewards = np.array(rewards)
        for step in range(self._window_size, len(np_rewards)):
            window = np_rewards[step - self._window_size:step]
            average_value = np.sum(window, axis=0) / self._window_size
            average_rewards.append(average_value)

        average_rewards = np.squeeze(np.stack(average_rewards), axis=-1)

        # self._run_measurement_manager.measurements.add_custom_data('cumulative_rewards', cumulative_reward)
        self._run_measurement_manager.measurements.add_custom_data('average_rewards', average_rewards)


def partition_runs(data: Sequence, n_partitions: int):
    iterator = iter(data)
    return list(iter(lambda: list(islice(iterator, n_partitions)), []))


@dataclass
class TimeSeries:
    mins: np.ndarray
    maxes: np.ndarray
    stddevs: np.ndarray
    means: np.ndarray


def means_mins_maxes(data):
    mins = np.min(data, axis=0)
    maxes = np.max(data, axis=0)
    stddevs = np.std(data, axis=0)
    means = np.mean(data, axis=0)

    return TimeSeries(mins, maxes, stddevs, means)


def multi_means_mins_maxes(data):
    means: List[List[float]] = [ts.means.tolist() for ts in data]
    mins: List[List[float]] = [ts.mins.tolist() for ts in data]
    maxes: List[List[float]] = [ts.maxes.tolist() for ts in data]

    return means, mins, maxes


class GoalDirectedTemplate(ExperimentTemplateBase[GoalDirectedTemplateTopology]):
    def __init__(self, name, avg_reward_window_size):
        super().__init__(name, avg_reward_window_size=avg_reward_window_size)
        self._avg_reward_window_size = avg_reward_window_size

    def setup_controller(self, topology: TTopology, controller: ExperimentController,
                         run_measurement_manager: RunMeasurementManager):
        controller.register(GoalDirectedTemplateMainComponent(topology, run_measurement_manager,
                                                              window_size=self._avg_reward_window_size))

    def publish_results(self, document: Document, docs_folder: str, measurement_manager: MeasurementManager,
                        topology_parameters: List[str]):
        steps = measurement_manager.get_values_from_all_runs('current_step')
        labels = topology_parameters

        multi_config_rewards = measurement_manager.get_custom_data_from_all_runs('average_rewards')

        for single_config_multi_run_rewards in multi_config_rewards:
            num_runs = len(single_config_multi_run_rewards[0])
            single_runs = [[] for _ in range(num_runs)]
            for timestep in single_config_multi_run_rewards:
                for i in range(num_runs):
                    single_runs[i].append(timestep[i])

            max_value = max(max(run) for run in single_runs)

            for i, timeseries in enumerate(single_runs):
                title = f"average_reward_run_{i}"
                plot_multiple_runs(steps,
                                   timeseries,
                                   ylabel=f"average_reward_run_{i}",
                                   xlabel="steps",
                                   ylim=[0, max_value],
                                   labels=labels,
                                   title=title,
                                   path=path.join(docs_folder, title),
                                   doc=document)

            means = []
            mins = []
            maxes = []
            for timestep in single_config_multi_run_rewards:
                means.append(np.mean(timestep))
                mins.append(np.min(timestep))
                maxes.append(np.max(timestep))

            title = "all_average_rewards"
            plot_multiple_runs(steps,
                               means,
                               y_lower=mins,
                               y_upper=maxes,
                               ylabel="average_reward",
                               xlabel="steps",
                               labels=labels,
                               title=title,
                               path=path.join(docs_folder, title),
                               doc=document)
