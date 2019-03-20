from eval_utils import parse_test_args, run_just_model
from torchsim.core.models.expert_params import SamplingMethod
from torchsim.core.nodes.dataset_phased_se_objects_task_node import SeObjectsTaskPhaseParams
from torchsim.research.experiment_templates.task0_train_test_template_relearning import Task0TrainTestTemplateRelearning
from torchsim.research.research_topics.rt_2_1_1_relearning.adapters.task0_relearn_basic_adapter import \
    Task0RelearnBasicAdapter
from torchsim.research.research_topics.rt_2_1_1_relearning.topologies.task0_basic_topology_phased import \
    SeT0BasicTopologyRT211Phased

DEF_MAX_STEPS = 200

TRAINING_PHASE_STEPS = 8000
TESTING_PHASE_STEPS = 2000
NUM_TRAINING_PHASES_DIV_2 = 4
NUM_TRAINING_PHASES = NUM_TRAINING_PHASES_DIV_2 * 2


def read_max_steps(max_steps: int = None):
    if max_steps is None:
        return DEF_MAX_STEPS
    return max_steps


def run_measurement_task0(name, params, args, topology_class, run_labels,
                          learning_rate=SeT0BasicTopologyRT211Phased.LEARNING_RATE):
    """Runs the experiment with specified params, see the parse_test_args method for arguments."""
    experiment = Task0TrainTestTemplateRelearning(
        Task0RelearnBasicAdapter(),
        topology_class,
        params,
        run_labels=run_labels,
        overall_training_steps=NUM_TRAINING_PHASES * TRAINING_PHASE_STEPS,
        num_testing_phases=NUM_TRAINING_PHASES,
        num_testing_steps=TESTING_PHASE_STEPS,
        measurement_period=1,
        learning_rate=learning_rate,
        save_cache=args.save,
        load_cache=args.load,
        clear_cache=args.clear,
        experiment_name=name,
        computation_only=args.computation_only,
        results_folder=args.alternative_results_folder
    )

    if args.run_gui:
        run_just_model(topology_class(**params[0]), gui=True)
    else:
        print(f'======================================= Measuring model: {name}')
        experiment.run()


def extend_with_phase_information(params, testing_class_filter=None, testing_location_filter=None):
    """add switching between training and testing, as parametrized in the beginning of the file

    params contain class_filter and location_filter - remove them, move them to testing
    """
    testing_class_filter_was_none = testing_class_filter is None
    testing_location_filter_was_none = testing_location_filter is None

    for param_dict in params:
        class_filter = param_dict['class_filter']
        location_filter = param_dict['location_filter']
        if testing_class_filter_was_none:
            testing_class_filter = class_filter
        if testing_location_filter_was_none:
            testing_location_filter = location_filter
        del param_dict['class_filter']
        del param_dict['location_filter']
        param_dict['phase_params'] = []
        for i in range(0, NUM_TRAINING_PHASES):
            phase_train = SeObjectsTaskPhaseParams(class_filter=class_filter, is_training=True,
                                                   num_steps=TRAINING_PHASE_STEPS, location_filter=location_filter)
            phase_test = SeObjectsTaskPhaseParams(class_filter=testing_class_filter, is_training=False,
                                                  num_steps=TESTING_PHASE_STEPS, location_filter=testing_location_filter)
            param_dict['phase_params'].append(phase_train)
            param_dict['phase_params'].append(phase_test)

    return params


def extend_with_class_filter_params(params):
    class_filters = [
        [1, 2],
        [1, 2, 3, 4],
        [1, 2, 3, 4, 5, 6],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0],
    ]
    new_params = _extend_params_with_param(params, class_filters, 'class_filter')
    return new_params, ["class filter " + str(len(x)) for x in class_filters]


def extend_with_location_filter_params(params):
    location_filters = [
        0.1,
        0.2,
        0.4,
        0.6,
        0.8,
        1.0,
    ]
    new_params = _extend_params_with_param(params, location_filters, 'location_filter')
    return new_params, ["location filter " + str(x) for x in location_filters]


def extend_with_cluster_center_params(params):
    cluster_centers = [
        1000,
        2000,
        3000,
        5000,
    ]
    new_params = _extend_params_with_param(params, cluster_centers, 'num_ccs')
    return new_params, ["num. ccs " + str(x) for x in cluster_centers]


def extend_with_sampling_method_params(params):
    sampling_methods = [
        SamplingMethod.LAST_N,
        SamplingMethod.BALANCED,
    ]
    new_params = _extend_params_with_param(params, sampling_methods, 'sampling_method')
    return new_params, [str(x) for x in sampling_methods]


def extend_with_buffer_size_params(params):
    buffer_sizes = [
        1000,
        2000,
        4000,
        10000,
    ]
    new_params = _extend_params_with_param(params, buffer_sizes, 'buffer_size')
    return new_params, ["buffer size " + str(x) for x in buffer_sizes]


def _extend_params_with_param(params, extending_params, extending_param_name):
    new_params = []
    for param in params:
        for extending_param in extending_params:
            new_param = param.copy()
            new_param[extending_param_name] = extending_param
            new_params.append(new_param)
    return new_params


def run_measurements_for_task0(custom_durations: bool = False):
    """Runs just the SPFlock on SEDatasetNode."""
    args = parse_test_args()

    # curriculum = (0, -1)  # we're using dataset 0

    # *******************************************************
    # stable data, different params (~ hyperparameter search)

    # - class filter
    # - location filter
    # - number of cluster centers
    # - buffer size
    # - sampling strategy
    lf = 1.0
    ccs = 200
    cf = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

    default_params_a = [{'location_filter': lf, 'num_ccs': ccs}]
    params_class_filter, run_labels_class_filter = extend_with_class_filter_params(default_params_a)
    extend_with_phase_information(params_class_filter)

    default_params_b = [{'class_filter': cf, 'num_ccs': ccs}]
    params_location_filter, run_labels_location_filter = extend_with_location_filter_params(default_params_b)
    extend_with_phase_information(params_location_filter)

    default_params_c = [{'class_filter': cf, 'location_filter': lf,  'sampling_method': SamplingMethod.BALANCED,
                         'buffer_size': 5000}]
    params_cluster_centers, run_labels_cluster_centers = extend_with_cluster_center_params(default_params_c)
    extend_with_phase_information(params_cluster_centers)

    default_params_d = [{'class_filter': cf, 'location_filter': lf, 'num_ccs': ccs}]
    params_buffer_size, run_labels_buffer_size = extend_with_buffer_size_params(default_params_d)
    extend_with_phase_information(params_buffer_size)

    default_params_e = [{'class_filter': cf, 'location_filter': lf, 'num_ccs': ccs}]
    params_sampling_method, run_labels_sampling_method = extend_with_sampling_method_params(default_params_e)
    extend_with_phase_information(params_sampling_method)

    topology = SeT0BasicTopologyRT211Phased

    run_measurement_task0("EXPERIMENT WITH CLASS FILTER", params_class_filter, args, topology, run_labels_class_filter)

    run_measurement_task0("EXPERIMENT WITH LOCATION FILTER", params_location_filter, args, topology,
                          run_labels_location_filter)

    run_measurement_task0("EXPERIMENT WITH NUMBER OF CLUSTER CENTERS", params_cluster_centers, args, topology,
                          run_labels_cluster_centers)

    run_measurement_task0("EXPERIMENT WITH BUFFER SIZE", params_buffer_size, args, topology, run_labels_buffer_size)

    run_measurement_task0("EXPERIMENT WITH BUFFER SAMPLING METHOD", params_sampling_method, args, topology,
                          run_labels_sampling_method)

    # ************************************************
    # shifting data, different ways how the data shift
    topology = SeT0BasicTopologyRT211Phased

    # abrupt shift of data
    cl_a = [6, 7, 8, 9, 10]
    cl_b = [1, 2, 3, 4, 5]
    cl_c = [11, 12, 13, 14, 15]
    phase_param1 = SeObjectsTaskPhaseParams(class_filter=cl_a + cl_b, is_training=True, num_steps=TRAINING_PHASE_STEPS, location_filter=1.0)
    phase_param2 = SeObjectsTaskPhaseParams(class_filter=cl_a + cl_c, is_training=True, num_steps=TRAINING_PHASE_STEPS, location_filter=1.0)
    phase_test_a = SeObjectsTaskPhaseParams(class_filter=cl_b, is_training=False, num_steps=TESTING_PHASE_STEPS, location_filter=1.0)
    phase_test_b = SeObjectsTaskPhaseParams(class_filter=cl_c, is_training=False, num_steps=TESTING_PHASE_STEPS, location_filter=1.0)

    first_part = [phase_param1, phase_test_a]
    second_part = [phase_param2, phase_test_a]
    phase_params = first_part * NUM_TRAINING_PHASES_DIV_2 + second_part * NUM_TRAINING_PHASES_DIV_2

    default_params_abrupt_a = [{'num_ccs': ccs, 'phase_params': phase_params}]

    run_measurement_task0("EXPERIMENT WITH ABRUPT DISTRIBUTION CHANGE A",
                          default_params_abrupt_a, args, topology, ["abrupt distr. change, old data"])

    first_part = [phase_param1, phase_test_b]
    second_part = [phase_param2, phase_test_b]
    phase_params = first_part * NUM_TRAINING_PHASES_DIV_2 + second_part * NUM_TRAINING_PHASES_DIV_2

    default_params_abrupt_b = [{'num_ccs': ccs, 'phase_params': phase_params}]
    run_measurement_task0("EXPERIMENT WITH ABRUPT DISTRIBUTION CHANGE B",
                          default_params_abrupt_b, args, topology, ["abrupt distr. change, new data"])

    # gradual shift of data
    cl_all = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    phase_params = []
    for i in range(0, NUM_TRAINING_PHASES_DIV_2):
        # train with narrow location data and measure
        first_part = SeObjectsTaskPhaseParams(class_filter=cl_all, is_training=True, num_steps=TRAINING_PHASE_STEPS,
                                              location_filter=0.2)
        second_part = SeObjectsTaskPhaseParams(class_filter=cl_all, is_training=False, num_steps=TESTING_PHASE_STEPS,
                                               location_filter=0.2)
        phase_params.append(first_part)
        phase_params.append(second_part)

    for i in range(0, NUM_TRAINING_PHASES_DIV_2):
        # then test re-learning with increasing range of location data
        location_filter = round(0.2 + (0.8 * (i+1) / NUM_TRAINING_PHASES_DIV_2), 2)
        first_part = SeObjectsTaskPhaseParams(class_filter=cl_all, is_training=True, num_steps=TRAINING_PHASE_STEPS,
                                              location_filter=location_filter)
        second_part = SeObjectsTaskPhaseParams(class_filter=cl_all, is_training=False, num_steps=TESTING_PHASE_STEPS,
                                               location_filter=0.2)
        phase_params.append(first_part)
        phase_params.append(second_part)

    default_params_gradual = [{'num_ccs': ccs, 'phase_params': phase_params}]

    run_measurement_task0("EXPERIMENT WITH GRADUAL DISTRIBUTION CHANGE",
                          default_params_gradual, args, topology, ["gradual distr. change, locations"])

    # note: we're using test data from the dataset in the testing phases


if __name__ == '__main__':
    run_measurements_for_task0(custom_durations=True)
