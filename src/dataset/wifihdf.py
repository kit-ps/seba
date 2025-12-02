from .hdf import HdfDataset, HdfDataPoint

import h5py
import numpy as np
import copy
import torch
from torch.utils.data import Dataset


class SingleTorchWifiDataset(Dataset):
    def __init__(self, x, y, class_to_idx, idx_to_class):
        self.x = x
        self.y = y
        self.class_to_idx = class_to_idx
        self.idx_to_class = idx_to_class

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.as_tensor(self.x[idx], dtype=torch.float32)
        return x, self.class_to_idx[self.y[idx]]

class DoubleTorchWifiDataset(Dataset):
    def __init__(self, x1, x2, y, class_to_idx, idx_to_class):
        self.x1 = x1
        self.x2 = x2
        self.y = y
        self.class_to_idx = class_to_idx
        self.idx_to_class = idx_to_class

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x1 = torch.as_tensor(self.x1[idx], dtype=torch.float32)
        x2 = torch.as_tensor(self.x2[idx], dtype=torch.float32)

        if len(x1.shape) == 2:
            tmp = torch.cat((x1, x2), dim=1)
        else:
            tmp = torch.cat((x1, x2), dim=3)
        return torch.flatten(tmp, start_dim=1), self.class_to_idx[self.y[idx]]

class WifihdfDataPoint(HdfDataPoint):
    def cut_or_pad(self, length):
        for key, values in self.data.items():
            if len(values) > length:
                self.data[key] = values[:length]
            elif len(values) < length:
                self.data[key] = np.pad(values, [(0, length - len(values)), (0, 0)], 'constant')

    def remove_ts(self):
        for key, values in self.data.items():
            self.data[key] = np.delete(values, 0, -1)


class WifihdfDataset(HdfDataset):
    pointclass = WifihdfDataPoint

    def load_hdf(self, file):
        with h5py.File(file, 'r') as f:
            for pid, sessions in f.items():
                for sid, actions in sessions.items():
                    for aname, adata in actions.items():
                        if 'action' not in self.config or aname == self.config['action']:
                            if aname == 'empty':
                                self.load_action(pid, sid, aname, None, adata)
                            elif aname == 'move':
                                for wname, wdata in adata.items():
                                    if 'wstyle' not in self.config or wname == self.config['wstyle']:
                                        self.load_action(pid, sid, aname, wname, wdata)
                            elif aname == 'pose':
                                pass # TODO
                            else:
                                raise ValueError("HDF File has unexpected actions.")

        if len(self.points) < 1:
            raise AttributeError("no points matching filters have been found!")
        self.calculate_norms()

    def load_action(self, pid, sid, aname, wname, wdata):
        for artname, artifacts in wdata.items():
            if 'artifact' not in self.config or artname == self.config['artifact']:
                for mname, machines in artifacts.items():
                    if 'machine' not in self.config or mname == 'bfid{}'.format(self.config['machine']):
                        if 'sequence_lengths' in machines.keys():
                            data = {}
                            for key, values in machines.items():
                                if key != 'sequence_lengths':
                                    data[key] = np.split(values, np.cumsum(machines['sequence_lengths'])[:-1])
                            for point_no in range(len(list(data.values())[0])):
                                point_meta = {
                                    'identity': pid,
                                    'session': sid[4:],
                                    'action': aname,
                                    'style': wname,
                                    'artifact': artname,
                                    'machine': mname[4:],
                                    'pointname': point_no
                                }
                                point_data = {}
                                for key, value in data.items():
                                    point_data[key] = value[point_no]
                                self.points.append(self.pointclass(point_meta, point_data))

    def calculate_norms(self):
        keys = self.points[0].data.keys()
        self.stds = {}
        self.means = {}
        for key in keys:
            data = list(map(lambda x: x.data[key], self.points))
            for i in data:
                np.nan_to_num(i, nan=0.0, copy=False)
            self.stds[key] = np.concatenate(data).std(axis=0)
            self.means[key] = np.concatenate(data).mean(axis=0)

    def copy(self, id_filter=None, sess_filter=None, point_filter=None, newname=None, softlinked=False):
        dataset = super().copy(id_filter=id_filter, sess_filter=sess_filter, point_filter=point_filter, newname=newname, softlinked=softlinked)
        dataset.stds = self.stds
        dataset.means = self.means
        return dataset

    def merge_and_rename(self, datasets, mappings, newname=None, softlinked=True):
        dataset = super().merge_and_rename(datasets, mappings, newname=newname, softlinked=softlinked)
        dataset.calculate_norms()
        return dataset

    def to_torch(self, keys, normalize=True, ts_removed=False):
        labels = copy.deepcopy(list(map(lambda x: x.label, self.points)))
        class_to_idx = {self.identities[i]: i for i in range(len(self.identities))}

        if len(keys) == 1:
            data = copy.deepcopy(list(map(lambda x: x.data[keys[0]], self.points)))
            if normalize:
                for i, e in enumerate(data):
                    if ts_removed:
                        e[:, :] -= self.means[keys[0]][:-1]
                        e[:, :] /= self.stds[keys[0]][:-1]
                    else:
                        e[:, :-1] -= self.means[keys[0]][:-1]
                        e[:, :-1] /= self.stds[keys[0]][:-1]
                    data[i] = e
            return SingleTorchWifiDataset(data, labels, class_to_idx, self.identities)
        elif len(keys) == 2:
            data0 = copy.deepcopy(list(map(lambda x: x.data[keys[0]], self.points)))
            data1 = copy.deepcopy(list(map(lambda x: x.data[keys[1]], self.points)))
            if normalize:
                for i, e in enumerate(data0):
                    e[:, :-1] -= self.means[keys[0]][:-1]
                    e[:, :-1] /= self.stds[keys[0]][:-1]
                    data0[i] = e
                for i, e in enumerate(data1):
                    e[:, :-1] -= self.means[keys[0]][:-1]
                    e[:, :-1] /= self.stds[keys[0]][:-1]
                    data1[i] = e
            return DoubleTorchWifiDataset(data0, data1, labels, class_to_idx, self.identities)

    def cut_or_pad(self, length):
        for point in self.points:
            point.cut_or_pad(length)

    def remove_ts(self):
        for point in self.points:
            point.remove_ts()

    def filter_points(self, pfilter):
        self.points = list(filter(pfilter, self.points))
