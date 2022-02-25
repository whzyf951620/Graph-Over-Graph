# from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data, InMemoryDataset
from torchvision.transforms import ToTensor
from TUdatasets import TUDataset
from utils import Indegree
import torch
import os
import pickle
import random
import configparser


BATCH_SIZE = 50
# dataset = TUDataset('data/%s'%DATA_TYPE, DATA_TYPE, pre_transform=Indegree(), use_node_attr=True)
# print(dataset.__dict__)
# NUM_FEATURES, NUM_CLASSES = dataset.num_features, dataset.num_classes


class Metalistic(InMemoryDataset):
    def __init__(self, config, split):
        super(Metalistic, self).__init__()
        self.data_type = config.get('NCI1', 'data_type')
        self.dataset = TUDataset('data/%s'%self.data_type, self.data_type, pre_transform=Indegree(), use_node_attr=True)
        self.NumClasses = int(self.dataset.num_classes)
        self.NumFeatures = int(self.dataset.num_node_features)
        self.TrainTestRatio = float(config.get(self.data_type, 'TrainTestRatio'))
        self.TrainvalRatio = float(config.get(self.data_type, 'TrainvalRatio'))
        self.savefiles = config.get(self.data_type, 'savefiles')
        if not os.path.exists(os.path.join(self.savefiles, self.data_type)):
            os.makedirs(os.path.join(self.savefiles, self.data_type))
        self.split = split

    def DataSplit(self):
        SplitFile = os.path.join(self.savefiles, self.data_type, 'splitIndex.pkl')
        if os.path.exists(SplitFile):
            pass
        else:
            GraphNum = self.dataset.slices['y'].shape[0] - 1
            Indiceslist = list(range(GraphNum))
            random.shuffle(Indiceslist)
            TestIndices = Indiceslist[:int(GraphNum * self.TrainTestRatio)]
            TrainvalIndices = Indiceslist[int(GraphNum * self.TrainTestRatio):]
            ValIndices = TrainvalIndices[:int(len(TrainvalIndices) * self.TrainvalRatio)]
            TrainIndices = TrainvalIndices[int(len(TrainvalIndices) * self.TrainvalRatio):]
            IndexDct = {'trainIndex': TrainIndices, 'valIndex': ValIndices, 'testIndex': TestIndices}
            with open(os.path.join(self.savefiles, self.data_type, 'splitIndex.pkl'), 'wb') as f:
                pickle.dump(IndexDct, f)
                f.close()

    def GetPklPath(self, split):
        FilePath = os.path.join(self.savefiles, self.data_type, split + '.pkl')
        return FilePath

    def DatasetPartition(self):
        DatasetPath = self.GetPklPath(self.split)
        SplitIndexFile = os.path.join(self.savefiles, self.data_type, 'splitIndex.pkl')
        f_Index = open(SplitIndexFile, 'rb')
        Key = self.split + 'Index'
        Index = pickle.load(f_Index)[Key]
        f_Index.close()
        if os.path.exists(DatasetPath):
            data = pickle.load(open(DatasetPath, 'rb'))
            return data
        else:
            Index = torch.as_tensor(Index, dtype=torch.long)
            data = self.dataset[Index]
            DataFile = os.path.join(self.savefiles, self.data_type, self.split + '.pkl')
            with open(DataFile, 'wb') as f:
                pickle.dump(data, f)
                f.close()
            return data

    def LblAndUnlblSplit(self):
        LblUnlblSplitFile = os.path.join(self.savefiles, self.data_type, self.split+'LblUnlbl.pkl')
        if os.path.exists(LblUnlblSplitFile):
            LblUnlblIndex = pickle.load(open(LblUnlblSplitFile, 'rb'))
            LblIndex = LblUnlblIndex['lbl']
            UnlblIndex = LblUnlblIndex['unlbl']
            return LblIndex, UnlblIndex

        else:
            f_LblUnlbl = open(LblUnlblSplitFile, 'wb')
            data = self.DatasetPartition()
            length = len(data._indices)
            Indices = list(range(length))
            # Indices = data._indices
            random.shuffle(Indices)
            LblNum = int(length * 0.4)

            LblUnlblIndex = {}
            LblIndex = Indices[:LblNum]
            UnlblIndex = Indices[LblNum:]
            LblUnlblIndex['lbl'] = LblIndex
            LblUnlblIndex['unlbl'] = UnlblIndex

            pickle.dump(LblUnlblIndex, f_LblUnlbl)
            f_LblUnlbl.close()

            return LblIndex, UnlblIndex

import numpy as np
import torch


class EposideSampler(object):
    '''
    PrototypicalBatchSampler: yield a batch of indexes at each iteration.
    Indexes are calculated by keeping in account 'classes_per_it' and 'num_samples',
    In fact at every iteration the batch indexes will refer to  'num_support' + 'num_query' samples
    for 'classes_per_it' random classes.
    __len__ returns the number of episodes per epoch (same as 'self.iterations').
    '''

    def __init__(self, labels, classes_per_it, num_samples, iterations):
        '''
        Initialize the PrototypicalBatchSampler object
        Args:
        - labels: an iterable containing all the labels for the current dataset
        samples indexes will be infered from this iterable.
        - classes_per_it: number of random classes for each iteration
        - num_samples: number of samples for each iteration for each class (support + query)
        - iterations: number of iterations (episodes) per epoch
        '''
        super(EposideSampler, self).__init__()
        self.labels = labels
        self.classes_per_it = classes_per_it
        self.sample_per_class = num_samples
        self.iterations = iterations

        self.classes, self.counts = np.unique(self.labels, return_counts=True)
        self.classes = torch.LongTensor(self.classes)

        # create a matrix, indexes, of dim: classes X max(elements per class)
        # fill it with nans
        # for every class c, fill the relative row with the indices samples belonging to c
        # in numel_per_class we store the number of samples for each class/row
        self.idxs = range(len(self.labels))
        self.indexes = np.empty((len(self.classes), max(self.counts)), dtype=int) * np.nan
        self.indexes = torch.Tensor(self.indexes)
        self.numel_per_class = torch.zeros_like(self.classes)
        for idx, label in enumerate(self.labels):
            label_idx = np.argwhere(self.classes == label).item()
            self.indexes[label_idx, np.where(np.isnan(self.indexes[label_idx]))[0][0]] = idx
            self.numel_per_class[label_idx] += 1

    def __iter__(self):
        '''
        yield a batch of indexes
        '''
        spc = self.sample_per_class
        cpi = self.classes_per_it

        for it in range(self.iterations):
            batch_size = spc * cpi
            batch = torch.LongTensor(batch_size)
            c_idxs = torch.randperm(len(self.classes))[:cpi]
            for i, c in enumerate(self.classes[c_idxs]):
                s = slice(i * spc, (i + 1) * spc)
                # FIXME when torch.argwhere will exists
                label_idx = torch.arange(len(self.classes)).long()[self.classes == c].item()
                sample_idxs = torch.randperm(self.numel_per_class[label_idx])[:spc]
                batch[s] = self.indexes[label_idx][sample_idxs]
            batch = batch[torch.randperm(len(batch))]
            yield batch

    def __len__(self):
        '''
        returns the number of iterations (episodes) per epoch
        '''
        return self.iterations


if __name__ == '__main__':
    config = configparser.ConfigParser()
    ConfigFilename = r'./config.ini'
    config.read(ConfigFilename)
    split = ['train', 'val', 'test']
    for sp in split:
        dataset = Metalistic(config, split=sp)
        dataset.DataSplit()
        dataset.DatasetPartition()
        dataset.LblAndUnlblSplit()


