import torch
import numpy as np
from torch.nn.functional import interpolate

from torchsim.core import get_float


class SampleCollectionOverseer:
    def __init__(self, render_width: int, render_height: int, num_train_trajectories: int, num_test_trajectories: int):

        self.render_width = render_width
        self.render_height = render_height
        self.num_train_trajectories = num_train_trajectories
        self.num_test_trajectories = num_test_trajectories

        self.train_images = []
        self.train_labels = []
        self.train_instance_ids = []
        self.train_examples_per_trajectory = []
        self.test_images = []
        self.test_labels = []
        self.test_instance_ids = []
        self.test_examples_per_trajectory = []

        self.last_label = None
        self.trajectories_count = 0
        self.examples = 0

    def add_sample(self, image, label, instance_id):
        # train samples
        if self.trajectories_count < self.num_train_trajectories:
            if not np.array_equal(label, self.last_label):
                self.train_examples_per_trajectory.append(self.examples)
                self.examples = 0
                self.trajectories_count += 1
                self.last_label = label
                print("Observation {0}".format(self.trajectories_count))

            if self.examples > 0:
                self.train_images.append(image)
                self.train_labels.append(label)
                self.train_instance_ids.append(instance_id)
            self.examples += 1
            print("Observation {0}".format(self.trajectories_count))

        # test samples
        elif self.trajectories_count < (self.num_train_trajectories + self.num_test_trajectories):
            if not np.array_equal(label, self.last_label):
                self.test_examples_per_trajectory.append(self.examples)
                self.examples = 0
                self.trajectories_count += 1
                self.last_label = label
                print("Observation {0}".format(self.trajectories_count))

            if self.examples > 0:
                self.test_images.append(image)
                self.test_labels.append(label)
                self.test_instance_ids.append(instance_id)
            self.examples += 1
        else:
            self.save_measurements('./data/eval/', 64, 64)
            self.save_measurements('./data/eval/', 32, 32)
            self.save_measurements('./data/eval/', 24, 24)

    def save_measurements(self, path: str, render_width: int, render_height):
        train_images = torch.cat(self.train_images, dim=0)
        train_labels = torch.cat(self.train_labels, dim=0)
        train_instance_ids = torch.from_numpy(np.array(self.train_instance_ids)).type(torch.long)
        train_examples_per_trajectory = torch.from_numpy(np.array(self.train_examples_per_trajectory)).type(torch.long)

        test_images = torch.cat(self.test_images, dim=0)
        test_labels = torch.cat(self.test_labels, dim=0)
        test_instance_ids = torch.from_numpy(np.array(self.test_instance_ids)).type(torch.long)
        test_examples_per_trajectory = torch.from_numpy(np.array(self.test_examples_per_trajectory)).type(torch.long)

        if self.render_width != render_width and self.render_height != render_height:
            train_images = interpolate(train_images.type(get_float(train_images.device)).unsqueeze(0),
                                       size=(render_width, render_height, 3)).squeeze(0).type(torch.uint8)
            test_images = interpolate(test_images.type(get_float(test_images.device)).unsqueeze(0),
                                      size=(render_width, render_height, 3)).squeeze(0).type(torch.uint8)

        train_data = [train_images,
                      train_labels,
                      train_instance_ids,
                      train_examples_per_trajectory]

        test_data = [test_images,
                     test_labels,
                     test_instance_ids,
                     test_examples_per_trajectory]

        # create a snippet containing 20 trajectories:
        last_id = torch.sum(train_examples_per_trajectory[:20])
        train_snippet = [train_images[:last_id].clone(),
                         train_labels[:last_id].clone(),
                         train_instance_ids[:last_id].clone(),
                         train_examples_per_trajectory[:20].clone()]

        last_id = torch.sum(test_examples_per_trajectory[:20])
        test_snippet = [test_images[:last_id].clone(),
                        test_labels[:last_id].clone(),
                        test_instance_ids[:last_id].clone(),
                        test_examples_per_trajectory[:20].clone()]

        task = "T0"
        size = str(render_width) + 'x' + str(render_height)

        torch.save(train_data, path + "SE_" + task + "_" + size + "_" + "train" + "_" + "full.set")
        torch.save(train_snippet, path + "SE_" + task + "_" + size + "_" + "train" + "_" + "snippet.set")

        torch.save(test_data, path + "SE_" + task + "_" + size + "_" + "test" + "_" + "full.set")
        torch.save(test_snippet, path + "SE_" + task + "_" + size + "_" + "test" + "_" + "snippet.set")
