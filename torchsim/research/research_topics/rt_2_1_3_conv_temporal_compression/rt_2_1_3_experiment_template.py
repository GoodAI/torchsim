import torch
from os import path
from typing import Any, Tuple, Union, List, Dict

from torchsim.core import get_float
from torchsim.core.eval.doc_generator.document import Document
from torchsim.core.eval.experiment_template_base import TestableExperimentTemplateBase, ExperimentTemplateBase
from torchsim.core.eval.measurement_manager import MeasurementManagerBase
from torchsim.core.eval.series_plotter import get_stamp, plot_multiple_runs, add_fig_to_doc, to_safe_name
from torchsim.core.eval.topology_adapter_base import TestableTopologyAdapterBase
from torchsim.core.nodes import ExpertFlockNode, SpatialPoolerFlockNode
from torchsim.core.utils.tensor_utils import detect_qualitative_difference
from torchsim.research.research_topics.rt_2_1_3_conv_temporal_compression.topologies.l3_conv_topology import L3ConvTopology, \
    L3SpConvTopology
from torchsim.utils.list_utils import flatten_list
from torchsim.utils.template_utils.template_helpers import compute_derivations, compute_classification_accuracy, \
    argmax_tensor, compute_se_classification_accuracy


class Rt213Adapter(TestableTopologyAdapterBase):
    _topology: Union[L3ConvTopology, L3SpConvTopology]

    device = 'cpu'
    dtype = get_float(device)

    def __init__(self):
        self._topology = None

    def get_topology(self) -> L3ConvTopology:
        return self._topology

    def set_topology(self, topology: L3ConvTopology):
        self._topology = topology

    def is_in_training_phase(self, **kwargs) -> bool:
        return self._topology.is_training

    def switch_to_training(self):
        self._topology.switch_tt(True)

    def switch_to_testing(self):
        self._topology.switch_tt(False)

    def l0_sp_execution_counter(self):
        return self._topology.conv_layers[0].expert_flock_nodes[0].memory_blocks.sp.execution_counter_forward \
            .tensor.clone().to(self.device).type(self.dtype)

    def l1_sp_execution_counter(self):
        return self._topology.conv_layers[1].expert_flock_nodes[0].memory_blocks.sp.execution_counter_forward \
            .tensor.clone().to(self.device).type(self.dtype)

    def l2_sp_execution_counter(self):
        return self._topology.sp_reconstruction_layer.sp_node.memory_blocks.sp.execution_counter_forward \
            .tensor.clone().to(self.device).type(self.dtype)

    def l0_output(self):
        return self._topology.conv_layers[0].outputs.data \
            .tensor.clone().to(self.device).type(self.dtype)

    def l1_output(self):
        return self._topology.conv_layers[1].outputs.data \
            .tensor.clone().to(self.device).type(self.dtype)

    def input_data(self):
        return self._topology.conv_layers[0].inputs.data \
            .tensor.clone().to(self.device).type(self.dtype)

    def l2_reconstructed_label(self):
        return self._topology.sp_reconstruction_layer.outputs.label \
            .tensor.clone().to(self.device).type(self.dtype)

    def correct_label(self):
        return self._topology.env_node.get_correct_label_memory_block() \
            .tensor.clone().to(self.device).type(self.dtype)

    def _get_sp_output(self, expert):
        if isinstance(expert, ExpertFlockNode):
            return expert.memory_blocks.sp.forward_clusters \
                .tensor.clone().to(self.device).type(self.dtype)
        elif isinstance(expert, SpatialPoolerFlockNode):
            return expert.outputs.sp.forward_clusters \
                .tensor.clone().to(self.device).type(self.dtype)
        else:
            raise ValueError(f"Unknown expert class {expert.__class__.__name__}")

    def l0_sp_output(self):
        expert = self._topology.conv_layers[0].expert_flock_nodes[0]
        return self._get_sp_output(expert).type(self.dtype)

    def l1_sp_output(self):
        expert = self._topology.conv_layers[1].expert_flock_nodes[0]
        return self._get_sp_output(expert).type(self.dtype)

    def l2_sp_output(self):
        expert = self._topology.sp_reconstruction_layer.sp_node
        return self._get_sp_output(expert).type(self.dtype)

    def get_n_classes(self):
        return self._topology.env_node.params.n_shapes


class Rt213ExperimentTemplate(TestableExperimentTemplateBase):
    _topology_adapter: Rt213Adapter

    def __init__(self,
                 adapter: Rt213Adapter,
                 topology_class,
                 models_params: List[Union[Tuple[Any], Dict[str, Any]]],
                 overall_training_steps: int,
                 num_testing_steps: int,
                 num_testing_phases: int,
                 sub_experiment_name: str,
                 computation_only: bool = False,
                 save_cache: bool = False,
                 load_cache: bool = False,
                 clear_cache=True,
                 experiment_folder=None):

        self.sub_experiment_name = sub_experiment_name

        super().__init__(adapter, topology_class, models_params, overall_training_steps, num_testing_steps,
                         num_testing_phases, computation_only=computation_only, save_cache=save_cache,
                         load_cache=load_cache, clear_cache=clear_cache, experiment_folder=experiment_folder)

        m_m = self._create_measurement_manager(self._experiment_folder, zip_data=True, delete_after_each_run=True)
        self._measurement_manager = m_m
        m_m.add_measurement_f('l0_sp_execution_counter', adapter.l0_sp_execution_counter)
        m_m.add_measurement_f('l1_sp_execution_counter', adapter.l1_sp_execution_counter)
        m_m.add_measurement_f('l2_sp_execution_counter', adapter.l2_sp_execution_counter)

        m_m.add_measurement_f('l0_output', adapter.l0_output)
        m_m.add_measurement_f('l1_output', adapter.l1_output)

        m_m.add_measurement_f('input_data', adapter.input_data)

        m_m.add_measurement_f('l0_sp_output', adapter.l0_sp_output)
        m_m.add_measurement_f('l1_sp_output', adapter.l1_sp_output)
        m_m.add_measurement_f('l2_sp_output', adapter.l2_sp_output)

        m_m.add_measurement_f('l2_reconstructed_label', adapter.l2_reconstructed_label)
        m_m.add_measurement_f('correct_label', adapter.correct_label)

        self.sp_executions_measurement_names = ['l0_sp_execution_counter', 'l1_sp_execution_counter',
                                                'l2_sp_execution_counter']

        self.sp_output_measurement_names = ['l0_sp_output', 'l1_sp_output',
                                            'l2_sp_output']

        self.sp_executions: List[List[List[float]]] = []
        self.sp_executions_dt: List[List[List[float]]] = []
        self.training_steps: List[float] = []

        self.classification_accuracy: List[List[float]] = []
        self.classification_accuracy_se: List[List[float]] = []

        self.sp_output_stability: List[List[List[float]]] = []

    def _get_measurement_manager(self) -> MeasurementManagerBase:
        return self._measurement_manager

    def _after_run_finished(self):
        last_measurement = self._measurement_manager.run_measurements[-1]

        if len(self.training_steps) == 0:
            training_step_in_training_phases = last_measurement.partition_to_list_of_training_phases(
                item_name='training_step',
                remove_steps=True)
            self.training_steps = flatten_list(training_step_in_training_phases)

        executions = []
        for measurement_name in self.sp_executions_measurement_names:
            execution = last_measurement.partition_to_list_of_training_phases(item_name=measurement_name,
                                                                              remove_steps=True)
            execution = flatten_list(execution)
            execution = [x.type(torch.float).mean().item() for x in execution]
            executions.append(execution)

        self.sp_executions.append(executions)

        classification_accuracy_run = []
        classification_accuracy_se_run = []

        for reconstructed_labels_test_phase, correct_labels_test_phase in \
                zip(last_measurement.partition_to_list_of_testing_phases('l2_reconstructed_label'),
                    last_measurement.partition_to_list_of_testing_phases('correct_label')):

            classification_accuracy_run.append(
                compute_classification_accuracy([argmax_tensor(x) for x in correct_labels_test_phase],
                                                [argmax_tensor(x) for x in reconstructed_labels_test_phase]))

            classification_accuracy_se_run.append(
                compute_se_classification_accuracy([argmax_tensor(x) for x in correct_labels_test_phase],
                                                   [argmax_tensor(x) for x in reconstructed_labels_test_phase],
                                                   num_classes=self._topology_adapter.get_n_classes()))

        self.classification_accuracy.append(classification_accuracy_run)
        self.classification_accuracy_se.append(classification_accuracy_se_run)

        sp_output_changes = []
        for measurement_name in self.sp_output_measurement_names:
            sp_outputs = last_measurement.partition_to_list_of_training_phases(item_name=measurement_name,
                                                                               remove_steps=True)
            sp_outputs = flatten_list(sp_outputs)
            tp_execution = [1.] + [detect_qualitative_difference(torch.flatten(x, 0, -2),
                                                                 torch.flatten(y, 0, -2)
                                                                 ).type(torch.float).mean().item()
                                   for x, y in zip(sp_outputs, sp_outputs[1:])]
            sp_output_changes.append(tp_execution)

        self.sp_output_stability.append(sp_output_changes)

    def _compute_experiment_statistics(self):
        pass

    def _publish_results(self):

        doc = Document()

        xlabel = "steps"
        ylabel = "number of SP forward executions"
        title_fe_dt = "smoothed derivation of SP forward executions and TP forward executions (SP O QD)"
        figsize = (18, 12)
        date = get_stamp()

        layer_names = ['L0', 'L1', 'L2']
        nr_layers = len(layer_names)
        labels = list(map(lambda x: x + " forward execution", layer_names)) + \
                 list(map(lambda x: x + " qualitative difference", layer_names))

        colors = ['b', 'orange', 'g', 'r', 'p']
        color_params = [{'c': color} for color in colors[:nr_layers]]
        color_ls_params = [{'c': color, 'ls': '--'} for color in colors[:nr_layers]]
        other_params = color_params + color_ls_params

        params_description = ExperimentTemplateBase.parameters_to_string(self._topology_parameters_list)
        for run in range(len(self.sp_executions)):
            sp_execution_dt = list(map(compute_derivations, self.sp_executions[run]))
            sp_output_dt = self.sp_output_stability[run]

            title = title_fe_dt + f" run {run} "
            fig = plot_multiple_runs(x_values=self.training_steps,
                                     y_values=sp_execution_dt + sp_output_dt,
                                     ylim=[0, 1],
                                     labels=labels,
                                     smoothing_window_size=501,
                                     xlabel=xlabel,
                                     ylabel=ylabel,
                                     title=title + params_description[run],
                                     figsize=figsize,
                                     hide_legend=False,
                                     other_params=other_params
                                     )
            add_fig_to_doc(fig, path.join(self._docs_folder, to_safe_name(title)), doc)

        title = "classification accuracy from reconstructed labels"
        fig = plot_multiple_runs(x_values=list(range(self._num_testing_phases)),
                                 y_values=self.classification_accuracy,
                                 ylim=[0, 1],
                                 labels=params_description,
                                 smoothing_window_size=None,
                                 xlabel="accuracy",
                                 ylabel="phases",
                                 title=title,
                                 figsize=figsize,
                                 hide_legend=False
                                 )
        add_fig_to_doc(fig, path.join(self._docs_folder, to_safe_name(title)), doc)

        title = "SE classification accuracy from reconstructed labels"
        fig = plot_multiple_runs(x_values=list(range(self._num_testing_phases)),
                                 y_values=self.classification_accuracy_se,
                                 ylim=[0, 1],
                                 labels=params_description,
                                 smoothing_window_size=None,
                                 xlabel="SE accuracy",
                                 ylabel="phases",
                                 title=title,
                                 figsize=figsize,
                                 hide_legend=False
                                 )
        add_fig_to_doc(fig, path.join(self._docs_folder, to_safe_name(title)), doc)

        doc.write_file(path.join(self._docs_folder, f"main.html"))

    def _experiment_template_name(self):
        return "RT_2_1_3_" + self.sub_experiment_name
