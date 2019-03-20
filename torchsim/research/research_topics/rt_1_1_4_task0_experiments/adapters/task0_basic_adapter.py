from torchsim.research.research_topics.rt_1_1_4_task0_experiments.adapters.task0_adapter_base import Task0AdapterBase


class Task0BasicAdapter(Task0AdapterBase):

    def get_title(self) -> str:
        return 'T0 - Basic topology'

    def is_output_id_available_for(self, layer_id: int) -> bool:
        """Expects that all expert flocks have flock_size=1"""
        return True
