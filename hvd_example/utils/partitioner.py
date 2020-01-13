# Dataset partitioning helper
import random
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class Partition(object):

    def __init__(self, data, index) :
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object) :

    def __init__(self, data, sizes=[0.7, 0.2, 0.1]):
        self.data = data
        self.partitions = []
        data_len = len(data)
        indexes = [x for x in range(data_len)]
        random.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]
        
    def use(self, partition):
        return Partition(self.data, self.partitions[partition])



        