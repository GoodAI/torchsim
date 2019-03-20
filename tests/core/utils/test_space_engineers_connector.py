import unittest
from typing import List

import numpy as np

from torchsim.utils.space_engineers_connector import SpaceEngineersConnector


class TestSpaceEngineersConnector(unittest.TestCase):

    def test_list_of_bools_to_bitmap(self):
        def do(list: List[bool]) -> int:
            return SpaceEngineersConnector.list_of_bools_to_bitmap(list)

        self.assertEqual(0x0005, do([True, False, True]))
        self.assertEqual(0x0001, do([True]))
        self.assertEqual(0x0003, do([True, True]))
        self.assertEqual(0x0006, do([False, True, True]))
        self.assertEqual(0x000c, do([False, False, True, True]))
        self.assertEqual(0x0018, do([False, False, False, True, True]))
