from typing import Generator, List, Any

import torch

from torchsim.core.graph.worker_node_base import WorkerNodeBase
from torchsim.core.nodes.motion_detection_node import MotionDetectionNode, MotionDetectionParams
from tests.core.nodes.node_unit_test_base import NodeTestBase


class TestMotionDetectionNodeGrayScale(NodeTestBase):

    @classmethod
    def setup_class(cls, device: str = 'cuda'):
        super().setup_class()

        cls.sy = 10
        cls.sx = 5
        cls.num_channels = 1

    def _generate_input_tensors(self) -> Generator[List[torch.Tensor], None, None]:

        zeros = self._creator.zeros([self.sy, self.sx, self.num_channels],
                                    dtype=torch.float,
                                    device=self._creator.device)
        image = self._creator.zeros_like(zeros)

        image[0, 0] = 0.1
        image[3, 2] = 0.33
        image[7, 4] = 0.99

        # show some sequence of inputs
        yield [zeros, image, image, image, zeros]

    def _generate_expected_results(self) -> Generator[List[Any], None, None]:

        zeros = self._creator.zeros([self.sy, self.sx],
                                    dtype=torch.float,
                                    device=self._creator.device)
        saliency = self._creator.zeros_like(zeros)

        saliency[0, 0] = 0.1
        saliency[3, 2] = 0.33
        saliency[7, 4] = 0.99

        # detect the absolute differences (except the first step, since it is disabled by default)
        yield [zeros, saliency, zeros, zeros, saliency]

    def _create_node(self) -> WorkerNodeBase:
        return MotionDetectionNode()


class TestMotionDetectionNode(NodeTestBase):

    @classmethod
    def setup_class(cls, device: str = 'cuda'):
        super().setup_class()

        cls.sy = 10
        cls.sx = 5
        cls.num_channels = 3

        cls.r = 0.2126
        cls.g = 0.7152
        cls.b = 0.0722

    def _generate_input_tensors(self) -> Generator[List[torch.Tensor], None, None]:

        image = self._creator.zeros([self.sy, self.sx, self.num_channels],
                                    dtype=torch.float,
                                    device=self._creator.device)
        image[0, 0, 0] = 0.1
        image[3, 2, 1] = 0.33
        image[7, 4, 0] = 0.99

        yield [image, 2*image]

    def _generate_expected_results(self) -> Generator[List[Any], None, None]:
        zeros = self._creator.zeros([self.sy, self.sx], dtype=torch.float, device=self._creator.device)
        saliency = self._creator.zeros_like(zeros)

        saliency[0, 0] = 0.1 * self.r
        saliency[3, 2] = 0.33 * self.g
        saliency[7, 4] = 0.99 * self.b

        yield [zeros, saliency]

    def _create_node(self) -> WorkerNodeBase:
        return MotionDetectionNode()


class TestMotionDetectionNodeThresholded(NodeTestBase):

    @classmethod
    def setup_class(cls, device: str = 'cuda'):
        super().setup_class()

        cls.sy = 11
        cls.sx = 5
        cls.num_channels = 3

        cls.threshold_value = 0.025

        cls.r = 0.2126
        cls.g = 0.7152
        cls.b = 0.0722

    def _generate_input_tensors(self) -> Generator[List[torch.Tensor], None, None]:

        zeros = self._creator.zeros([self.sy, self.sx, self.num_channels],
                                    dtype=torch.float,
                                    device=self._creator.device)
        image = self._creator.zeros_like(zeros)

        image[0, 0, 0] = 0.1
        image[3, 2, 1] = 0.33
        image[7, 4, 0] = 0.99

        yield [zeros, image, image, zeros, image, zeros, zeros]

    def _generate_expected_results(self) -> Generator[List[Any], None, None]:
        zeros = self._creator.zeros([self.sy, self.sx], dtype=torch.float, device=self._creator.device)
        saliency = self._creator.zeros_like(zeros)

        vr = 0.1 * self.r
        vg = 0.33 * self.g
        vb = 0.99 * self.b

        saliency[0, 0] = 0 if vr < self.threshold_value else vr
        saliency[3, 2] = 0 if vg < self.threshold_value else vg
        saliency[7, 4] = 0 if vb < self.threshold_value else vb

        yield [zeros, saliency, zeros, saliency, saliency, saliency, zeros]

    def _create_node(self) -> WorkerNodeBase:
        params = MotionDetectionParams()
        params.use_thresholding = True
        params.threshold_value = self.threshold_value
        return MotionDetectionNode()

