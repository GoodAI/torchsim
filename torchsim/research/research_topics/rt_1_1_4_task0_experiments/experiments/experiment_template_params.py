class ExperimentTemplateParamsBase:
    def __init__(self,
                 measurement_period: int = 5,
                 sliding_window_size: int = 300,
                 sliding_window_stride: int = 100,
                 sp_evaluation_period: int = 200):
        self.measurement_period = measurement_period
        self.sliding_window_size = sliding_window_size
        self.sliding_window_stride = sliding_window_stride
        self.sp_evaluation_period = sp_evaluation_period


class ExperimentTemplateParams(ExperimentTemplateParamsBase):
    """
    Parameters of the ExperimentTemplate,
    which should be changed between debug/full version of the experiment.
    """

    def __init__(self,
                 measurement_period: int = 5,
                 sliding_window_size: int = 300,
                 sliding_window_stride: int = 100,
                 sp_evaluation_period: int = 200,
                 max_steps: int = 70000):

        super().__init__(measurement_period, sliding_window_size, sliding_window_stride, sp_evaluation_period)

        self.max_steps = max_steps


class TrainTestExperimentTemplateParams(ExperimentTemplateParamsBase):

    def __init__(self,
                 measurement_period: int = 5,
                 sliding_window_size: int = 300,
                 sliding_window_stride: int = 100,
                 sp_evaluation_period: int = 200,

                 overall_training_steps: int = 70000,
                 num_testing_steps: int = 500,
                 num_testing_phases: int = 20):

        super().__init__(measurement_period, sliding_window_size, sliding_window_stride, sp_evaluation_period)

        self.overall_training_steps = overall_training_steps
        self.num_testing_steps = num_testing_steps
        self.num_testing_phases = num_testing_phases
