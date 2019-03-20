from torchsim.research.research_topics.rt_1_1_4_task0_experiments.adapters.task0_adapter_base import Task0AdapterBase


class Task0NarrowAdapter(Task0AdapterBase):

    def is_output_id_available_for(self, layer_id: int) -> bool:
        """All the topology has flock_size=1"""
        return True

    def get_title(self) -> str:
        return 'T0 - Narrow hierarchy'

