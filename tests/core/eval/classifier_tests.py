import pytest
import torch

from torchsim.core.eval.metrics.simple_classifier_nn import ClassifierHiddenNet, ClassifierTrainer, \
    ClassifierNoHiddenNet, compute_nn_classifier_accuracy, BatchSampler
from torchsim.core.eval.metrics.simple_classifier_svm import compute_svm_classifier_accuracy


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_forward_pass(device):
    """Test correct dimensions and softmaxing of the outputs."""

    input_size = 10
    hidden_size = 11
    output_size = 12
    n_samples = 3

    tolerance = 0.0001  # outputs of the network (softmax) are not accurate

    net = ClassifierHiddenNet(input_size, hidden_size, output_size, device)

    net.training = False
    input = torch.rand(n_samples, input_size, device=device)
    output = net.forward(input)

    # correct output shape
    assert output.shape[0] == input.shape[0]
    assert output.shape[1] == output_size

    # assert the softmax computed across the correct dimension
    sum_outputs = output.sum(dim=1)
    assert sum_outputs.shape[0] == n_samples and sum_outputs.numel() == n_samples
    sum_outputs.to('cpu')

    for sample in range(0, n_samples):
        output = sum_outputs[sample].item()
        assert 1.0 + tolerance > output > 1.0 - tolerance

    # check the softmax is not applied during training
    net.training = True
    output2 = net.forward(input)
    assert output2.sum() != n_samples * 1.0


def generate_data(n_classes, n_samples, noise_amplitude, device: str):
    """Generates simple noisy one-hot vectors and corresponding labels."""

    # random labels
    labels = torch.randint(n_classes, [n_samples], device=device).unsqueeze(1)

    # noisy one-hot vectors generated from the labels
    inputs = torch.rand(n_samples, n_classes, device=device) * noise_amplitude
    ones = torch.ones([n_samples], device=device).unsqueeze(1)
    inputs.scatter_add_(dim=1, index=labels.long(), src=ones)

    labels = labels.squeeze().long()

    return inputs, labels


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_trivial_training_task_hidden_layer(device):
    """Test the ClassifierTrainer: training and the stopping criterion."""
    n_classes = 9
    n_samples = 100
    noise_amplitude = 0.1

    learn_rate = 0.1

    inputs, labels = generate_data(n_classes, n_samples, noise_amplitude, device)

    # define and train the classifier
    net = ClassifierHiddenNet(input_size=n_classes, hidden_size=n_classes, n_classes=n_classes, device=device)
    trainer = ClassifierTrainer(net,
                                device=device,
                                learn_rate=learn_rate,
                                max_loss_difference=0.0005, log_loss=False)
    trainer.train(inputs, labels)

    acc = trainer.compute_accuracy(inputs, labels)
    assert acc > 0.9


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_trivial_training_task_no_hidden_layer(device):
    """Test the same for the classifier without the hidden layer."""

    n_classes = 7
    n_samples = 31
    noise_amplitude = 0.1
    learn_rate = 0.1

    inputs, labels = generate_data(n_classes, n_samples, noise_amplitude, device)

    net = ClassifierNoHiddenNet(input_size=n_classes, n_classes=n_classes, device=device)
    trainer = ClassifierTrainer(net, device=device, learn_rate=learn_rate, max_loss_difference=0.0005)

    trainer.train(inputs, labels)

    acc = trainer.compute_accuracy(inputs, labels)

    assert acc > 0.8


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_metric_works(device):
    """Tests the final product: given inputs and labels, compute the accuracy ~ quality of the inputs."""

    n_classes = 5
    n_samples = 19
    noise_amplitude = 0.1

    inputs, labels = generate_data(n_classes, n_samples, noise_amplitude, device)

    accuracy = compute_nn_classifier_accuracy(inputs, labels, n_classes)

    # print(f'accuracy is {accuracy}')
    assert accuracy > 0.8


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_metric_does_not_work_when_should_not(device):
    """Labels do not depend on the data, metric should return some small value."""

    n_classes = 7
    n_samples = 300

    # generate labels and random data for them
    _, labels = generate_data(n_classes, n_samples, 1, device)
    inputs = torch.rand(n_samples, n_classes, device=device)

    accuracy = compute_nn_classifier_accuracy(inputs, labels, n_classes, log_loss=False)

    print(f'measured accuracy on random data is {accuracy}')
    assert accuracy < 0.3


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_metric_works_on_ids(device):
    """You can choose if the inputs will be n-dimensional vectors, or IDs."""

    n_classes = 3
    n_samples = 111

    _, labels = generate_data(n_classes, n_samples, 1, device)

    accuracy = compute_nn_classifier_accuracy(labels.clone(), labels, n_classes, custom_input_length=n_classes)

    # print(f'accuracy is {accuracy}')
    assert accuracy > 0.9


@pytest.mark.parametrize('device', ['cpu', 'cuda'],)
def test_metric_works_on_ids_different_than_no_classes(device):
    """Input in the format of scalars, but their range is different than n_classes."""

    input_multiplier = 4

    n_classes = 6
    n_samples = 111

    _, labels = generate_data(n_classes, n_samples, 1, device)

    inputs = labels * input_multiplier
    input_size = n_classes * input_multiplier

    accuracy = compute_nn_classifier_accuracy(inputs, labels, n_classes, custom_input_length=input_size)

    # print(f'accuracy is {accuracy}')
    assert accuracy > 0.9


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_metric_does_not_work_on_ids_when_should_not(device):
    """No mutual information between inputs and labels, classifier should be very bad."""
    n_classes = 13
    n_samples = 301

    # generate labels and random data for them

    labels = torch.randint(n_classes, [n_samples],device=device).long()
    different_labels = torch.randint(n_classes, [n_samples], device=device).long()

    accuracy = compute_nn_classifier_accuracy(different_labels, labels, n_classes, custom_input_length=n_classes)

    # print(f'measured accuracy on random data is {accuracy}')
    assert accuracy < 0.2


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_metric_works_on_ids_with_hidden_layer(device):
    """Test the version with the hidden layer."""

    n_classes = 3
    n_samples = 111

    _, labels = generate_data(n_classes, n_samples, 1, device)

    accuracy = compute_nn_classifier_accuracy(labels.clone(),
                                              labels,
                                              n_classes=n_classes,
                                              custom_input_length=n_classes,
                                              use_hidden_layer=True,
                                              log_loss=False)

    # print(f'accuracy is {accuracy}')
    assert accuracy > 0.6  # empirically set to this value


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_high_dimensional_input(device):
    """Classifier should work on high-dimensional inputs as well."""

    n_classes = 15
    n_samples = 300

    noise_amplitude = 0.1
    dimension_multiplier = 11   # the input is 11 times bigger than it would have to be

    # random labels
    labels = torch.randint(n_classes, [n_samples], device=device).unsqueeze(1)

    # noisy one-hot vectors, which have much bigger dimension than necessary
    inputs = torch.rand(n_samples, n_classes * dimension_multiplier, device=device) * noise_amplitude

    ones = torch.ones([n_samples], device=device).unsqueeze(1)
    inputs.scatter_add_(dim=1, index=labels.long(), src=ones)

    labels = labels.squeeze().long()

    accuracy = compute_nn_classifier_accuracy(inputs, labels, n_classes,
                                              log_loss=False, max_loss_difference=0.0001,
                                              learn_rate=0.2, max_epochs=300)

    print(f'accuracy = {accuracy}')

    assert accuracy > 0.9


def tensors_equal(first: torch.Tensor, second: torch.Tensor) -> bool:
    """Return True if the two given tensors are equal (shapes and data)."""

    if len(first.shape) != len(second.shape):
        print('tensor shapes not the same')
        return False

    for dim in range(0, len(first.shape)):
        if first.shape[dim] != second.shape[dim]:
            return False

    diff = first - second
    return sum(diff) == 0


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_sampler(device):
    """Test the mechanism which samples data into the mini-batches."""

    n_classes = 3
    n_samples = 320
    noise_amp = 0.01
    batch_size = 32

    inputs, labels = generate_data(n_classes, n_samples, noise_amp, device)

    bs = BatchSampler(inputs, labels, device=device, batch_size=batch_size)

    batch_inputs, batch_labels = bs.sample_to_batch()

    assert batch_inputs.shape[0] == batch_size
    assert batch_inputs.shape[1] == inputs.shape[1]

    assert batch_labels.shape[0] == batch_size
    assert len(batch_labels.shape) == 1

    another_inputs, another_labels = bs.sample_to_batch()

    no_same_data = 0

    # should sample different things every time if possible
    for sample in range(0, another_inputs.shape[0]):
        if tensors_equal(another_inputs[sample, :], batch_inputs[sample, :]):
            no_same_data += 1

    assert no_same_data < 4  # should not happen by chance (4 same samples at same positions)

    sample_counts = 2

    # test that it samples 320/32=10 times
    while True:
        a, b = bs.sample_to_batch()

        if a is None:
            break

        sample_counts += 1

    assert sample_counts == 10


def test_overflow():

    num = 256
    a = torch.ones(num).float() * 2.0
    b = torch.ones(num).float() * 2.0

    comparison = a == b
    # sum_comparison = comparison.sum(dim=0)
    sum_comparison = torch.sum(comparison)  # sum(comparison) returns uint8!

    # print(f'\n\nSum of comparisons is {sum_comparison} and should be {num},\n'
    #       f'a.dtype {a.dtype}, b.dtype {b.dtype}, sum dtype {sum_comparison.dtype}\n'
    #       f'version: {torch.__version__}\n')

    assert sum_comparison == num  # passes only for values <= 255


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('labels', [[0, 2, 0, 1], [[1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]]])
@pytest.mark.parametrize('inputs', [[1, 2, 3, 1], [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 1, 0, 0]]])
def test_svm_classifier(device, inputs, labels):
    """Tests the final product: given inputs and labels, compute the accuracy ~ quality of the inputs."""

    inputs = torch.tensor(inputs)
    labels = torch.tensor(labels)
    if len(inputs.shape) >= 2:
        custom_input_length = None
    else:
        custom_input_length = 5
    accuracy = compute_svm_classifier_accuracy(inputs, labels, 3, custom_input_length=custom_input_length)
    assert accuracy == 0.75

