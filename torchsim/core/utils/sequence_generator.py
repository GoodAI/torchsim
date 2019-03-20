import copy
from typing import List
import numpy as np


def diagonal_transition_matrix(length: int, diag_value: float = 0.9):
    """
    Creates numpy matrix like
    0.9  0.05 0.05
    0.05 0.9  0.05
    0.05 0.05 0.9

    :param length: width and height of matrix
    :param diag_value: value on main diagonal
    :return:
    """
    non_diag_val = (1. - diag_value) / (length - 1)
    return np.eye(length) * (diag_value - non_diag_val) + non_diag_val


class SequenceGenerator:
    """Iterator for probabilistic integer sequences.

    Generate probabilistic sequences, in the manner of the BrainSim sequence world. The iterator produces numbers listed
    in :_seqs:. After a seq is completed, a new one is selected according to :_transition_probs:.
    """
    _seqs: List[List[int]]
    _transition_probs: np.array
    _seq_index: int
    _position: int

    _random: np.random.RandomState

    def __init__(self, seqs: List[List[int]], transition_probs: np.array, random: np.random.RandomState = None):
        self._seqs = seqs
        self._transition_probs = transition_probs
        self._seq_index = 0
        self._position = -1
        self._random = random or np.random.RandomState()

    def __iter__(self):
        return self

    def __next__(self):
        self.advance()
        return self.peek()

    def advance(self):
        last_position = len(self._seqs[self._seq_index]) - 1
        if self._position == last_position:
            self._switch_seq()
            self._position = 0
        else:
            self._position = self._position + 1

    def peek(self):
        return self._seqs[self._seq_index][self._position]

    def current_sequence(self):
        return self._seq_index

    def _switch_seq(self):
        probs = self._transition_probs[self._seq_index]
        self._seq_index = self._sample_from(probs)
        self._position = 0

    def _sample_from(self, probs) -> int:
        acc = np.array(probs).cumsum()
        r = self._random.uniform()
        return int(np.argmax(acc >= r))

    @classmethod
    def from_length(cls, length: int, random: np.random.RandomState = None):
        return cls.from_list(list(range(length)), random)

    @classmethod
    def from_list(cls, seq: List[int], random: np.random.RandomState = None):
        return cls([seq], np.ones((1, 1)), random)

    @classmethod
    def from_multiple(cls, seqs: List[List[int]], transition_probs: np.matrix, random: np.random.RandomState = None):
        return cls(seqs, transition_probs, random)

    @property
    def current_sequence_id(self) -> int:
        return self._seq_index

    @property
    def transition_probs(self) -> np.array:
        return self._transition_probs

    @transition_probs.setter
    def transition_probs(self, value: np.array):
        self._transition_probs = value

    @staticmethod
    def default_transition_probs(seqs: List[List[int]]) -> np.array:
        number_of_seqs = len(seqs)
        uniform_prob = 1 / number_of_seqs
        return np.full((number_of_seqs, number_of_seqs), uniform_prob)

    def clone(self):
        return copy.deepcopy(self)
