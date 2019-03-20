import os
import pickle
import torch
import numpy

NUM_SNIPPETS = 50


class DatasetConverter:
    @staticmethod
    def convert_pickles(path: str):
        dir_list = os.listdir(path)
        for dir_file in dir_list:

            if '.pkl' in dir_file:
                print("Opening {0}".format(dir_file))
                pkl = open(path + '/' + dir_file, 'rb')
                header = pickle.load(pkl)
                dataset = pickle.load(pkl)
                pkl.close()

                properties = dir_file.split('_')
                task = properties[1]
                if task is "T0":
                    size = properties[3]
                    type = properties[4].split('.')[0]
                else:
                    size = properties[3].split('.')[0]
                    type = "train"

                print("Converting...")
                torch_data = []
                torch_snippet = []

                if not isinstance(dataset[0], torch.Tensor):
                    # convert to torch tensors
                    images = torch.from_numpy(numpy.array(dataset[0])).permute(0, 2, 3, 1)
                    labels = torch.from_numpy(numpy.array(dataset[1]))
                else:
                    images = dataset[0]
                    labels = dataset[1]

                torch_data.append(images)
                torch_data.append(labels)
                part_images = images[:NUM_SNIPPETS].clone()
                part_labels = labels[:NUM_SNIPPETS].clone()
                torch_snippet.append(part_images)
                torch_snippet.append(part_labels)

                torch.save(torch_data, path + '/' + "SE_" + task + "_" + size + "_" + type + "_" + "full.set")
                torch.save(torch_snippet, path + '/' + "SE_" + task + "_" + size + "_" + type + "_" + "snippet.set")

                os.remove(path + '/' + dir_file)

    @staticmethod
    def change_snippet_size(path: str):
        dir_list = os.listdir(path)
        for dir_file in dir_list:
            if '.set' in dir_file and 'snippet' in dir_file:
                print("Removed.")
                os.remove(path + dir_file)

        for dir_file in dir_list:
            if '.set' in dir_file:
                print("Opening {0}".format(dir_file))
                dataset = torch.load(path + dir_file)

                properties = dir_file.split('_')
                task = properties[1]
                size = properties[2]
                type = properties[3]
                part = properties[4].split('.')[0]

                print("Converting...")
                images = dataset[0]
                labels = dataset[1]
                instance_ids = dataset[2]
                examples = dataset[3]

                # create a snippet containing NUM_SNIPPETS trajectories:
                last_id = torch.sum(examples[:NUM_SNIPPETS])
                torch_snippet = [images[:last_id].clone(),
                                 labels[:last_id].clone(),
                                 instance_ids[:last_id].clone(),
                                 examples[:NUM_SNIPPETS].clone()]

                torch.save(torch_snippet, path + "SE_" + task + "_" + size + "_" + type + "_" + "snippet.set")


if __name__ == '__main__':
    DatasetConverter.change_snippet_size('./data/eval/')
    print("Done.")
