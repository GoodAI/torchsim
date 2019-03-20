import numpy as np
import pytest
import torch

from torchsim.core.memory.tensor_creator import AllocatingCreator
from torchsim.core.models.expert_params import ExpertParams, SamplingMethod
from torchsim.core.models.spatial_pooler import SPFlock, ConvSPFlock
from torchsim.core import FLOAT_NAN, get_float, FLOAT_TYPE_CPU
from torchsim.core.utils.tensor_utils import same
from tests.core.models.spatial_pooler.test_process import get_subflock_creation_testing_flock
from tests.testing_utils import copy_or_fill_tensor


class TestSPFlock:
    def test_forward_and_learning(self):
        def manually_calc_cc_targets():
            buffer = flock.buffer
            buffer_data = torch.full((flock.flock_size, flock.batch_size, flock.n_cluster_centers), dtype=float_dtype,
                                     fill_value=FLOAT_NAN,
                                     device=device)
            buffer.inputs._sample_batch(flock.batch_size, buffer_data)
            perm_data = buffer_data

            all_indices = torch.arange(flock.flock_size, dtype=torch.int64, device=flock._device)

            learn_process = flock._create_learning_process(all_indices)

            buffer_clusters, _ = learn_process.compute_closest_cluster_centers(flock.cluster_centers, buffer_data)
            perm_clusters = buffer_clusters

            all_sum_data = []
            # compute for each flock independently
            for exp_data, exp_clust in zip(perm_data, perm_clusters):
                sum_data = []
                for c in range(flock.n_cluster_centers):
                    indices = exp_clust.type(torch.ByteTensor)[:, c]
                    individual_data_points = exp_data[indices, :]
                    if sum(indices) == 0:
                        mean_data_points = torch.full((flock.input_size,), dtype=float_dtype, fill_value=FLOAT_NAN,
                                                      device=device)
                    else:
                        mean_data_points = torch.mean(individual_data_points, dim=0)
                    sum_data.append(mean_data_points)
                all_sum_data.append(torch.stack(sum_data))

            ground_truth = torch.stack(all_sum_data)
            return ground_truth

        def run_forward(data):
            forward_process = flock._create_forward_process(data, all_indices)
            forward_process.run_and_integrate()

        def run_learning():
            learn_process = flock._create_learning_process(all_indices)
            learn_process.run_and_integrate()

        n_cluster_centers = 2
        f_size = 3
        i_size = 2
        buff_size = 10
        ba_size = 1
        device = 'cuda'
        float_dtype = get_float(device)

        params = ExpertParams()
        params.flock_size = f_size
        params.n_cluster_centers = n_cluster_centers

        params.spatial.input_size = i_size
        params.spatial.buffer_size = buff_size
        params.spatial.batch_size = ba_size
        params.spatial.sampling_method = SamplingMethod.LAST_N

        flock = SPFlock(params, AllocatingCreator(device))
        flock.cluster_boosting_durations.fill_(0)

        all_indices = torch.arange(f_size, dtype=torch.int64, device=device).unsqueeze(dim=1)

        flock.cluster_centers = torch.tensor([[[0.25, 1.1], [-0.3, 0.2]],
                                              [[1.2, 1.3], [-1, -2]],
                                              [[0.6, 0.2], [-1.2, -0.8]]], dtype=float_dtype, device=device)

        # First pass - no learning happens
        data1 = torch.tensor([[0.2, 1], [1.1, 1.1], [-1.2, -0.7]], dtype=float_dtype, device=device)
        cc1 = torch.tensor([[1., 0], [1., 0], [0, 1]], dtype=float_dtype, device=device)

        run_forward(data1)
        cc_res1 = flock.forward_clusters

        run_learning()

        assert same(cc1, cc_res1)

        # Second pass - learning happens for experts 1 and 2
        data2 = torch.tensor([[0.2, 1], [-1, -0.4], [0.2, 0.3]], dtype=float_dtype, device=device)
        cc2 = torch.tensor([[1., 0], [0, 1.], [1, 0]], dtype=float_dtype, device=device)

        run_forward(data2)
        cc_res2 = flock.forward_clusters

        cc_targets2 = manually_calc_cc_targets()
        cc_deltas2 = (cc_targets2 - flock.cluster_centers)
        cc_deltas2[torch.isnan(cc_deltas2)] = 0

        run_learning()

        assert same(cc2, cc_res2)
        assert same(cc_targets2, flock.cluster_center_targets)
        assert same(cc_deltas2, flock.cluster_center_deltas)

        # Third pass, all inputs are valid, but only expert 0 learns
        data3 = torch.tensor([[0.1, 1], [-1.1, -0.4], [0.3, 0.3]], dtype=float_dtype, device=device)
        cc3 = torch.tensor([[1., 0], [0, 1.], [1, 0]], dtype=float_dtype, device=device)

        run_forward(data3)
        cc_res3 = flock.forward_clusters

        cc_targets3 = manually_calc_cc_targets()
        cc_deltas3 = (cc_targets3 - flock.cluster_centers)
        cc_deltas3[torch.isnan(cc_deltas3)] = 0

        run_learning()

        assert same(cc3, cc_res3)
        assert same(cc_targets3, flock.cluster_center_targets)
        assert same(cc_deltas3, flock.cluster_center_deltas)

        # Test with NaN input...
        copy_of_buffer = flock.buffer.inputs.stored_data.clone()

        data4 = torch.tensor([[0.1, 1], [-1.1, FLOAT_NAN], [0.3, 0.3]], dtype=float_dtype, device=device)
        cc4 = cc3

        run_forward(data4)
        cc_res4 = flock.forward_clusters

        run_learning()

        assert same(cc4, cc_res4), "Classification should still work with NaN in the input"

        data5 = torch.tensor([[FLOAT_NAN, 1], [-1, -0.4], [0.2, 0.3]], dtype=float_dtype, device=device)
        cc5 = cc2

        run_forward(data5)
        cc_res5 = flock.forward_clusters

        run_learning()

        assert same(cc5, cc_res5), "Classification should still work with NaN in the input"

        # Commented out because NanNs during learning are not currently handled.
        # more_nans = torch.tensor([[FLOAT_NAN, FLOAT_NAN], [FLOAT_NAN, FLOAT_NAN], [FLOAT_NAN, FLOAT_NAN]],
        #                         dtype=float_dtype, device=device)
        # nan_inputs = [data5, more_nans] * 3
        # for nan_input in nan_inputs:
        #     run_forward(nan_input)
        #     run_learning()
        #
        # run_forward(data4)
        # assert same(cc4, flock.forward_clusters), "Classification should stay the same after (not) learning with NaN input"
        # assert same(copy_of_buffer, flock.buffer.inputs.stored_data), "NaN input should not affect the buffer"

    @pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.slow)])
    @pytest.mark.parametrize('sampling_method',
                             [SamplingMethod.LAST_N, SamplingMethod.BALANCED, SamplingMethod.UNIFORM])
    @pytest.mark.parametrize('flock_class', [SPFlock, ConvSPFlock])
    def test_reconstruct(self, sampling_method, flock_class, device):
        float_dtype = get_float(device)

        params = ExpertParams()
        params.flock_size = 2
        params.n_cluster_centers = 4

        params.spatial.input_size = 5
        params.spatial.buffer_size = 7
        params.spatial.batch_size = 6

        params.spatial.sampling_method = sampling_method
        flock = flock_class(params, AllocatingCreator(device))

        flock.forward_clusters = torch.tensor([[0, 0, 1, 0],
                                               [0.2, 0.3, 0.4, 0.1]], dtype=float_dtype, device=device)

        flock.predicted_clusters = torch.tensor([[0, 0.5, 0.5, 0],
                                                 [1, 0, 0, 0]], dtype=float_dtype, device=device)

        flock.cluster_centers = torch.tensor([[[1, 0, 0, 0, 0],
                                               [0, 1, 0, 0, 0],
                                               [0, 0, 0.5, 0.5, 0],
                                               [0, 0, 0.5, 0, 0.5]],
                                              [[1, 0, 0, 0, 0],
                                               [0, 1, 0, 0, 0],
                                               [0, 0, 1, 0, 0],
                                               [0, 0, 0, 1, 0]]], dtype=float_dtype, device=device)

        flock.reconstruct(indices=None)

        expected_reconstructed_input = torch.tensor([[0, 0, 0.5, 0.5, 0],
                                                     [0.2, 0.3, 0.4, 0.1, 0]], dtype=float_dtype, device=device)

        expected_predicted_reconstructed_input = torch.tensor([[0, 0.5, 0.25, 0.25, 0],
                                                               [1, 0, 0, 0, 0]], dtype=float_dtype, device=device)

        assert same(expected_reconstructed_input, flock.current_reconstructed_input)
        assert same(expected_predicted_reconstructed_input, flock.predicted_reconstructed_input)

    @pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.slow)])
    @pytest.mark.parametrize('flock_class', [SPFlock, ConvSPFlock])
    def test_reconstruct_subflock(self, device, flock_class):
        float_dtype = get_float(device)

        params = ExpertParams()
        params.flock_size = 2
        params.n_cluster_centers = 4

        params.spatial.input_size = 5
        params.spatial.buffer_size = 7
        params.spatial.batch_size = 6

        flock = flock_class(params, AllocatingCreator(device))

        flock.forward_clusters = torch.tensor([[0, 0, 1, 0],
                                               [0, 0, 0, 0]], dtype=float_dtype, device=device)

        flock.predicted_clusters = torch.tensor([[0, 0.5, 0.5, 0],
                                                 [0, 0, 0, 0]], dtype=float_dtype, device=device)

        flock.cluster_centers = torch.tensor([[[1, 0, 0, 0, 0],
                                               [0, 1, 0, 0, 0],
                                               [0, 0, 0.5, 0.5, 0],
                                               [0, 0, 0.5, 0, 0.5]],
                                              [[1, 0, 0, 0, 0],
                                               [0, 1, 0, 0, 0],
                                               [0, 0, 1, 0, 0],
                                               [0, 0, 0, 1, 0]]], dtype=float_dtype, device=device)

        indices = torch.tensor([0], dtype=torch.int64, device=device)

        flock.reconstruct(indices)

        expected_reconstructed_input = torch.tensor([[0, 0, 0.5, 0.5, 0],
                                                     [0, 0, 0, 0, 0]], dtype=float_dtype, device=device)

        expected_predicted_reconstructed_input = torch.tensor([[0, 0.5, 0.25, 0.25, 0],
                                                               [0, 0, 0, 0, 0]], dtype=float_dtype, device=device)

        assert same(expected_reconstructed_input, flock.current_reconstructed_input)
        assert same(expected_predicted_reconstructed_input, flock.predicted_reconstructed_input)

    # Disabled because non-default streams do not work right now
    @pytest.mark.slow
    @pytest.mark.flaky(reruns=3)
    @pytest.mark.parametrize('sampling_method',
                             [SamplingMethod.LAST_N, SamplingMethod.BALANCED, SamplingMethod.UNIFORM])
    @pytest.mark.parametrize('use_default_stream', [True, pytest.param(False, marks=pytest.mark.skip(
        reason="We don't use non-default streams now"))])
    def test_forward_learn_streams(self, use_default_stream, sampling_method):
        params = ExpertParams()
        params.flock_size = 1
        params.n_cluster_centers = 4

        params.spatial.input_size = 2
        params.spatial.cluster_boost_threshold = 2
        params.spatial.learning_rate = 0.1
        params.spatial.learning_period = 1
        params.spatial.batch_size = 4
        params.spatial.buffer_size = 6
        params.spatial.max_boost_time = 10
        params.spatial.sampling_method = sampling_method

        device = 'cuda'
        float_dtype = get_float(device)

        flock = SPFlock(params, AllocatingCreator(device))

        data = torch.tensor([[0., 0], [1., 0], [0., 1], [1, 1]], dtype=float_dtype, device=device)

        iters = 20

        def run():
            for itr in range(iters):
                for k in data:
                    flock.forward_learn(k.view(1, -1))

        if use_default_stream:
            run()
        else:
            with torch.cuda.stream(torch.cuda.Stream()):
                run()

        expected_cluster_centers = data

        # Cluster centers have no guarantee of order - so we have to order them manually
        rounded_cluster_centers = np.around(flock.cluster_centers.cpu().data.numpy()[0], decimals=2)
        cc_indices = []
        for cc in rounded_cluster_centers:
            cc_indices.append(cc[0] * 2 + cc[1] * 4)

        sorted_indices = np.argsort(cc_indices)
        sorted_ccs = flock.cluster_centers[0, sorted_indices, :]

        assert same(expected_cluster_centers, sorted_ccs, eps=5e-2)

    @pytest.mark.slow
    @pytest.mark.flaky(reruns=5)
    @pytest.mark.parametrize('sampling_method',
                             [SamplingMethod.LAST_N, SamplingMethod.BALANCED, SamplingMethod.UNIFORM])
    def test_forward_learn_subflocking(self, sampling_method):
        params = ExpertParams()
        params.flock_size = 2
        params.n_cluster_centers = 4

        params.spatial.input_size = 2
        params.spatial.cluster_boost_threshold = 2
        params.spatial.learning_rate = 0.1
        params.spatial.learning_period = 1
        params.spatial.batch_size = 4
        params.spatial.buffer_size = 8
        params.spatial.max_boost_time = 10
        params.spatial.sampling_method = sampling_method

        device = 'cuda'
        float_dtype = get_float(device)

        flock = SPFlock(params, AllocatingCreator(device))

        # Flock1 sequence is [0, 0], [1, 0], [0, 1], [1, 1], [0, 1]
        # Flock2 sequence is [0, 0], [1, 0], [0, 1], [1, 1], [1, 1]
        data = torch.tensor(
            [[[0., 0], [0., 0]], [[1., 0], [1., 0]], [[0., 1], [0., 1]], [[1, 1], [1, 1]], [[0, 1], [1, 1]]],
            dtype=float_dtype, device=device)

        iters = 15

        for itr in range(iters):
            for k in data:
                flock.forward_learn(k)

        expected_forward_executions = torch.tensor([[iters * 5], [iters * 4]], dtype=torch.int64, device=device)
        expected_learning_executions = expected_forward_executions - (params.spatial.batch_size - 1)

        assert same(expected_forward_executions, flock.execution_counter_forward)
        assert same(expected_learning_executions, flock.execution_counter_learning)

        cluster_centers1 = flock.cluster_centers.to('cpu').type(FLOAT_TYPE_CPU).data[0].numpy()
        cluster_centers2 = flock.cluster_centers.to('cpu').type(FLOAT_TYPE_CPU).data[1].numpy()
        data = data.to('cpu').type(FLOAT_TYPE_CPU).data.numpy()
        expected_cluster_centers1 = data[:4, 0]
        expected_cluster_centers2 = data[:4, 1]

        # Cluster centers have no guarantee of order - so we have to order them manually
        rounded_cluster_centers1 = np.around(cluster_centers1, decimals=2)
        rounded_cluster_centers2 = np.around(cluster_centers2, decimals=2)

        cc_indices1 = []
        cc_indices2 = []
        for cc1, cc2 in zip(rounded_cluster_centers1, rounded_cluster_centers2):
            cc_indices1.append(cc1[0] * 2 + cc1[1] * 4)
            cc_indices2.append(cc2[0] * 2 + cc2[1] * 4)

        sorted_indices1 = np.argsort(cc_indices1)
        sorted_ccs1 = cluster_centers1[sorted_indices1]

        sorted_indices2 = np.argsort(cc_indices2)
        sorted_ccs2 = cluster_centers2[sorted_indices2]

        np.testing.assert_almost_equal(expected_cluster_centers1, sorted_ccs1, decimal=1)
        np.testing.assert_almost_equal(expected_cluster_centers2, sorted_ccs2, decimal=1)

    @pytest.mark.parametrize('enable_learning', [True, False])
    @pytest.mark.parametrize('SP_type', [SPFlock, ConvSPFlock])
    def test_forward_learn_enable_learning(self, enable_learning, SP_type):
        device = 'cuda'
        float_dtype = get_float(device)

        params = ExpertParams()
        params.flock_size = 1
        params.n_cluster_centers = 4

        params.spatial.input_size = 2
        params.spatial.cluster_boost_threshold = 2
        params.spatial.learning_rate = 0.1
        params.spatial.learning_period = 1
        params.spatial.batch_size = 4
        params.spatial.max_boost_time = 10
        assert params.spatial.enable_learning  # True should be the default value
        params.spatial.enable_learning = enable_learning

        flock = SP_type(params, AllocatingCreator(device))

        data = torch.tensor([[0., 0], [1., 0], [0., 1], [1, 1]], dtype=float_dtype, device=device)
        initial_cluster_centers = flock.cluster_centers.clone()

        for input in data:
            flock.forward_learn(input.view(1, -1))

        # should be different if enable_learning == True
        assert (not same(initial_cluster_centers, flock.cluster_centers)) == enable_learning

    @pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.slow)])
    def test_inverse_projection(self, device):
        float_dtype = get_float(device)

        params = ExpertParams()
        params.flock_size = 2
        params.n_cluster_centers = 4

        params.spatial.input_size = 5
        params.spatial.buffer_size = 7
        params.spatial.batch_size = 6

        flock = SPFlock(params, AllocatingCreator(device))

        flock.cluster_centers = torch.tensor([[[1, 0, 0, 0, 0],
                                               [0, 1, 0, 0, 0],
                                               [0, 0, 0.5, 0.5, 0],
                                               [0, 0, 0.5, 0, 0.5]],
                                              [[1, 0, 0, 0, 0],
                                               [0, 1, 0, 0, 0],
                                               [0, 0, 1, 0, 0],
                                               [0, 0, 0, 1, 0]]], dtype=float_dtype, device=device)

        data = torch.tensor([[0, 0, 1, 0],
                             [0.2, 0.3, 0.4, 0.1]], dtype=float_dtype, device=device)

        result = flock.inverse_projection(data)

        expected_projection = torch.tensor([[0, 0, 0.5, 0.5, 0],
                                            [0.2, 0.3, 0.4, 0.1, 0]], dtype=float_dtype, device=device)

        assert same(expected_projection, result)

    @pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.slow)])
    def test_create_forward_process(self, device):
        flock_size = 10
        input_size = 5
        float_dtype = get_float(device)

        flock, indices = get_subflock_creation_testing_flock(flock_size=flock_size, input_size=input_size,
                                                             device=device)

        data = torch.rand((flock_size, input_size), dtype=float_dtype, device=device)

        process = flock._create_forward_process(data, indices)

        expected_data = data[indices]
        expected_forward_clusters = flock.forward_clusters[indices]
        expected_cluster_centers = flock.cluster_centers[indices]

        assert process._buffer == flock.buffer
        assert same(expected_data, process._data)
        assert same(expected_forward_clusters, process._forward_clusters)
        assert same(expected_cluster_centers, process._cluster_centers)

    def test_create_learning_process(self):
        flock, indices = get_subflock_creation_testing_flock()

        learning = flock._create_learning_process(indices)

        flocked_boosting_durations = flock.cluster_boosting_durations[indices]
        flocked_prev_boost = flock.prev_boosted_clusters[indices]
        flocked_boosting_tar = flock.boosting_targets[indices]

        assert same(flocked_boosting_durations, learning._cluster_boosting_durations)
        assert same(flocked_prev_boost, learning._prev_boosted_clusters)
        assert same(flocked_boosting_tar, learning._boosting_targets)

    @pytest.mark.parametrize("mask, total_data_written, data_since_last_sample, expected_learn_tensor",
                             [(torch.ones(7, dtype=torch.uint8), 0, 0, torch.zeros(7, dtype=torch.uint8)),
                              (torch.ones(7, dtype=torch.uint8), 5, 5, torch.zeros(7, dtype=torch.uint8)),
                              (torch.ones(7, dtype=torch.uint8), 11, 1, torch.zeros(7, dtype=torch.uint8)),
                              (torch.ones(7, dtype=torch.uint8), 15, 5, torch.ones(7, dtype=torch.uint8)),
                              (torch.zeros(7, dtype=torch.uint8), 15, 5, torch.zeros(7, dtype=torch.uint8)),
                              (torch.tensor([0, 1, 1, 0, 0, 0, 1], dtype=torch.uint8), 15, 5,
                               torch.tensor([0, 1, 1, 0, 0, 0, 1], dtype=torch.uint8)),
                              (torch.tensor([0, 1, 1, 1, 0, 1, 1], dtype=torch.uint8),
                               torch.tensor([11, 9, 11, 11, 15, 8, 100], dtype=torch.uint8),
                               torch.tensor([5, 6, 4, 8, 5, 1, 5], dtype=torch.uint8),
                               torch.tensor([0, 0, 0, 1, 0, 0, 1], dtype=torch.uint8))])
    @pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.slow)])
    def test_determine_learning(self, mask, total_data_written, data_since_last_sample, expected_learn_tensor, device):
        """TODO: one-line summary here.

        - No data stored,  insufficient learning period, so no learning can run
        - Correct learning period, but insufficient data for batch
        - Sufficient data, incorrect learning period
        - Sufficient data and learn period and all forwarded
        - Sufficient data and learn period, but none forwarded
        - Sufficient data and learn period, and some forward
        - Different combinations of all three
        """
        params = ExpertParams()
        params.flock_size = 7
        params.n_cluster_centers = 5
        mask = mask.to(device)
        expected_learn_tensor = expected_learn_tensor.to(device)

        s_pooler = params.spatial
        s_pooler.batch_size = 10
        s_pooler.learning_period = 5

        flock = SPFlock(params, creator=AllocatingCreator(device))
        copy_or_fill_tensor(flock.buffer.total_data_written, total_data_written)
        copy_or_fill_tensor(flock.buffer.data_since_last_sample, data_since_last_sample)
        learn_tensor = flock._determine_learning(mask)
        assert same(expected_learn_tensor, learn_tensor)


class TestConvSPFlock:
    def test_conv_forward(self):
        params = ExpertParams()
        params.n_cluster_centers = 10
        params.flock_size = 5
        params.spatial.input_size = 3
        params.spatial.buffer_size = 30
        device = 'cuda'
        float_dtype = get_float(device)

        creator = AllocatingCreator(device)

        sp_flock = ConvSPFlock(params, creator)

        input_data = torch.tensor([[2, 0, 4],
                                   [1, 0.3, -1],
                                   [2, 0.1, 0.5],
                                   [0.7, 0.9, 0.8],
                                   [2, 0, 4]], dtype=float_dtype, device=device)

        forward_mask = sp_flock.forward(input_data)

        expected_forward_mask = torch.tensor([1, 1, 1, 1, 1], dtype=torch.uint8, device=device)

        assert same(expected_forward_mask, forward_mask)

        expected_common_buffer = torch.full((1, params.spatial.buffer_size, params.spatial.input_size),
                                            fill_value=FLOAT_NAN, dtype=float_dtype, device=device)
        expected_common_buffer[0, 0] = input_data[0]
        expected_common_buffer[0, 1] = input_data[1]
        expected_common_buffer[0, 2] = input_data[2]
        expected_common_buffer[0, 3] = input_data[3]
        expected_common_buffer[0, 4] = input_data[4]

        assert same(expected_common_buffer, sp_flock.common_buffer.inputs.stored_data)
        assert sp_flock.forward_clusters.sum() == params.flock_size
        assert same(sp_flock.forward_clusters[0], sp_flock.forward_clusters[4])
        assert sp_flock.common_buffer.current_ptr == 4
        assert sp_flock.common_buffer.total_data_written == 5

        assert (sp_flock.buffer.current_ptr == 0).all()
        assert (sp_flock.buffer.total_data_written == 1).all()

        for member in range(params.flock_size):
            assert same(input_data[member], sp_flock.buffer.inputs.stored_data[member, 0])

    @pytest.mark.parametrize("mask, total_data_written, data_since_last_sample, expected_learn_tensor",
                             [(torch.ones(1, dtype=torch.uint8), 0, 0, torch.zeros(1, dtype=torch.uint8)),
                              (torch.ones(1, dtype=torch.uint8), 5, 5, torch.zeros(1, dtype=torch.uint8)),
                              (torch.ones(1, dtype=torch.uint8), 11, 1, torch.zeros(1, dtype=torch.uint8)),
                              (torch.ones(1, dtype=torch.uint8), 15, 5, torch.ones(1, dtype=torch.uint8)),
                              (torch.zeros(1, dtype=torch.uint8), 15, 5, torch.zeros(1, dtype=torch.uint8))])
    def test_determine_learning(self, mask, total_data_written, data_since_last_sample, expected_learn_tensor):
        """Test for the mechanism which determines if the convolutional spatial pooler should learn.

        Args:
              mask (torch.Tensor): Mask of which experts performed a forward pass.
              total_data_written (int): How much data has been written to the buffer.
              data_since_last_sample (int): How much data has been written since the last learning pass.
              expected_learn_tensor (torch.Tensor): What mask we expect for the learning pass.

        - No data stored,  insufficient learning period, so no learning can run
        - Correct learning period, but insufficient data for batch
        - Sufficient data, incorrect learning period
        - Sufficient data and learn period and all forwarded
        - Sufficient data and learn period, but none forwarded
        """
        device = 'cpu'
        params = ExpertParams()
        params.flock_size = 7
        params.n_cluster_centers = 5

        s_pooler = params.spatial
        s_pooler.batch_size = 10
        s_pooler.learning_period = 5

        flock = ConvSPFlock(params, AllocatingCreator(device))

        flock.common_buffer.total_data_written.fill_(total_data_written)
        flock.common_buffer.data_since_last_sample.fill_(data_since_last_sample)
        learn_tensor = flock._determine_learning(mask)

        assert same(expected_learn_tensor, learn_tensor)

    @pytest.mark.slow
    @pytest.mark.flaky(reruns=3)
    @pytest.mark.parametrize('sampling_method',
                             [SamplingMethod.LAST_N, SamplingMethod.BALANCED, SamplingMethod.UNIFORM])
    def test_forward_learn_subflocking(self, sampling_method):
        params = ExpertParams()
        params.flock_size = 2
        params.n_cluster_centers = 4

        params.spatial.input_size = 2
        params.spatial.cluster_boost_threshold = 2
        params.spatial.learning_rate = 0.1
        params.spatial.learning_period = 1
        params.spatial.batch_size = 8
        params.spatial.buffer_size = 10
        params.spatial.max_boost_time = 10
        params.spatial.sampling_method = sampling_method

        device = 'cuda'
        float_dtype = get_float(device)

        flock = ConvSPFlock(params, AllocatingCreator(device))

        # Flock1 sequence is [0, 0], [1, 0], [0, 1], [1, 1], [0, 1]
        # Flock2 sequence is [0, 0], [1, 0], [0, 1], [1, 1], [1, 1]
        data = torch.tensor(
            [[[0., 0], [0., 0]], [[1., 0], [1., 0]], [[0., 1], [0., 1]], [[1, 1], [1, 1]], [[0, 1], [1, 1]]],
            dtype=float_dtype, device=device)

        iters = 15

        for itr in range(iters):
            for k in data:
                flock.forward_learn(k)

        expected_forward_executions = torch.tensor([[iters * 5], [iters * 4]], dtype=torch.int64, device=device)
        expected_learning_executions = torch.tensor([[iters * 5]], dtype=torch.int64, device=device) - 3
        expected_learning_executions = expected_learning_executions.expand(params.flock_size, 1)

        assert same(expected_forward_executions, flock.execution_counter_forward)
        assert same(expected_learning_executions, flock.execution_counter_learning)

        cluster_centers1 = flock.cluster_centers.to('cpu').data[0].numpy()
        cluster_centers2 = flock.cluster_centers.to('cpu').data[1].numpy()
        data = data.to('cpu').data.numpy()
        expected_cluster_centers1 = data[:4, 0]
        expected_cluster_centers2 = data[:4, 1]

        # Cluster centers have no guarantee of order - so we have to order them manually
        rounded_cluster_centers1 = np.around(cluster_centers1, decimals=2)
        rounded_cluster_centers2 = np.around(cluster_centers2, decimals=2)

        cc_indices1 = []
        cc_indices2 = []
        for cc1, cc2 in zip(rounded_cluster_centers1, rounded_cluster_centers2):
            cc_indices1.append(cc1[0] * 2 + cc1[1] * 4)
            cc_indices2.append(cc2[0] * 2 + cc2[1] * 4)

        sorted_indices1 = np.argsort(cc_indices1)
        sorted_ccs1 = cluster_centers1[sorted_indices1]

        sorted_indices2 = np.argsort(cc_indices2)
        sorted_ccs2 = cluster_centers2[sorted_indices2]

        np.testing.assert_almost_equal(expected_cluster_centers1, sorted_ccs1, decimal=1)
        np.testing.assert_almost_equal(expected_cluster_centers2, sorted_ccs2, decimal=1)
