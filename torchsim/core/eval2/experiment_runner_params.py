from typing import NamedTuple


class SingleExperimentRunParams(NamedTuple):
    """Parameters for a single experiment run. This is a subset of the ExperimentRunnerParams."""
    load_cache: bool
    save_cache: bool
    calculate_statistics: bool
    max_steps: int
    save_model_after_run: bool


class ExperimentParams(NamedTuple):
    """Parameters for the set of experiment runs which comprise the whole experiment run."""
    max_steps: int  # If this is set to 0, the simulation will run until a component says it should stop.
    seed: int = None
    calculate_statistics: bool = True
    save_cache: bool = False
    load_cache: bool = False
    clear_cache: bool = False
    experiment_folder: str = None
    docs_folder: str = None
    delete_cache_after_each_run: bool = False
    zip_cache: bool = False
    save_models_after_run: bool = True

    def create_run_params(self) -> SingleExperimentRunParams:
        return SingleExperimentRunParams(self.load_cache, self.save_cache, self.calculate_statistics, self.max_steps,
                                         self.save_models_after_run)
