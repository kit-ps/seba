from .abstract import AbstractDataset, AbstractDataPoint

import os
import uuid


class HdfDataPoint(AbstractDataPoint):
    def __init__(self, meta, data):
        self.attr = meta
        self.data = data
        self.identity = meta['identity']
        self.session = meta['session']
        self.pointname = meta['pointname']
        self.label = self.identity


class HdfDataset(AbstractDataset):
    pointclass = HdfDataPoint

    def run(self, datasets):
        self.load_hdf(os.path.join(self.dir, self.name + ".hdf"))

        self.identities = sorted(set(map(lambda x: x.identity, self.points)))
        self.labels = self.identities
        self.sessions = sorted(set(map(lambda x: x.session, self.points)))

        self.log.info(
            "Loaded dataset {} ({} datapoints, {} identities, {} sessions)".format(self.name, len(self.points), len(self.identities), len(self.sessions))
        )
        return self

    def load_hdf(self, file):
        pass

    def copy(self, id_filter=None, sess_filter=None, point_filter=None, newname=None, softlinked=False):
        if not softlinked:
            raise NotImplementedError("HDF Dataset cannot be hardlink-copied.")
        if newname is None:
            newname = str(uuid.uuid4())
        dataset = type(self)({'dir': self.dir, 'name': newname}, self.seed, self.context)

        dataset.points = list(filter(point_filter, filter(sess_filter, filter(id_filter, self.points))))
        dataset.identities = sorted(set(map(lambda x: x.identity, dataset.points)))
        dataset.labels = dataset.identities
        dataset.sessions = sorted(set(map(lambda x: x.session, dataset.points)))

        self.log.info(
            "Creating new dataset {} (copy of {}) (contains {} datapoints, {} identities, {} sessions)".format(newname, self.name, len(dataset.points), len(dataset.identities), len(dataset.sessions))
        )

        dataset.meta = {"name": newname, "original": self.name, "datatype": self.meta["datatype"]}
        if "original_meta" not in self.meta:
            dataset.meta["original_meta"] = self.meta
        else:
            dataset.meta["original_meta"] = self.meta["original_meta"]

        return dataset

    def merge_and_rename(self, datasets, mappings, newname=None, softlinked=True):
        # sanity checks
        if not len(set(map(lambda x: type(x), [self] + datasets))) == 1:
            self.log.error("Cannot merge dataests with different types!")
        if not len(datasets) + 1 == len(mappings):
            self.log.error("Merging: Mapping has invalid length!")
        a = [x for mapping in mappings for x in mapping.values()]
        if not len(a) == len(list(set(a))):
            self.log.error("Merging: Destination point names are not unique!")

        if not softlinked:
            raise NotImplementedError("HDF Dataset cannot be hardlink-copied.")
        if newname is None:
            newname = str(uuid.uuid4())
        newset = type(self)({'dir': self.dir, 'name': newname}, self.seed, self.context)

        # copy the points
        for ds, mapping in zip([self] + datasets, mappings):
            for point in ds[:]:
                id = "{}.{}.{}".format(point.identity, point.session, point.pointname)
                if id in mapping:
                    point.identity, point.session, point.pointname = mapping[id].split('.')
                    point.label = point.identity
                    newset.points.append(point)

        newset.identities = sorted(set(map(lambda x: x.identity, newset.points)))
        newset.labels = newset.identities
        newset.sessions = sorted(set(map(lambda x: x.session, newset.points)))

        self.log.info("Creating new dataset {} (merge of {}) (contains {} datapoints, {} identities, {} sessions)".format(newname, ", ".join(map(lambda x: x.name, [self] + datasets)), len(newset.points), len(newset.identities), len(newset.sessions)))

        # meta info stuff
        newset.meta = {"name": newname, "original": "|".join(map(lambda x: x.name, datasets)), "datatype": self.meta["datatype"]}
        if "original_meta" not in self.meta:
            newset.meta["original_meta"] = self.meta
        else:
            newset.meta["original_meta"] = self.meta["original_meta"]
        return newset

    def delete(self):
        raise NotImplementedError("Deleting strongly not recommended for HDF Datasets. Please do so manually!")

    def save_meta(self):
        pass
