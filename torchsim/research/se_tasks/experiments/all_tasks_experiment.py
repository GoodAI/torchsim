from torchsim.research.se_tasks.experiments.task_0_experiment import run_measurements_for_task0
from torchsim.research.se_tasks.experiments.task_1_experiment import run_measurements_for_task1

if __name__ == '__main__':
    """Runs the listed experiments."""

    custom_task_durations = True

    # ######### TASK 0 ###########
    run_measurements_for_task0(custom_durations=custom_task_durations)

    # ######### TASK 1 ###########
    run_measurements_for_task1(custom_durations=custom_task_durations)
