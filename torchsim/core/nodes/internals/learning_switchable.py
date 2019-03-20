from abc import abstractmethod, ABC


class LearningSwitchable(ABC):
    @abstractmethod
    def switch_learning(self, learning_on: bool):
        pass


class TestingSwitcher(ABC):
    @abstractmethod
    def is_learning(self) -> bool:
        pass
