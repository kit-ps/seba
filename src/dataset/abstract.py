import os.path
import logging
import yaml
import uuid
import os

class AbstractDataset:
    nout = 1

    def __init__(self, config, seed, context):
        self.log = logging.getLogger("seba.data")
        self.config = config
        self.seed = seed
        self.context = context

        self.points = []
        self.identities = []
        self.sessions = []

        if 'dir' in self.config:
            self.dir = self.config['dir']
        else:
            self.dir = os.path.join(os.getcwd(), "data")
        self.name = self.config['name']
        self.meta = self.load_meta()
        self.init()

    def init(self):
        pass

    def load_meta(self):
        metafile = os.path.join(self.dir, self.name + ".meta.yaml")
        if os.path.exists(metafile):
            with open(metafile, "r") as file:
                return yaml.load(file, Loader=yaml.SafeLoader)
        else:
            return {}

    def run(self, datasets):
        return self

    def __getitem__(self, key):
        def gen_id_filter(key):
            if key == slice(None, None, None):
                return None
            elif type(key) == str or type(key) == int:
                return (lambda x: x.identity == key)
            else:
                pass # TODO

        def gen_sess_filter(key):
            if key == slice(None, None, None):
                return None
            elif type(key) == str or type(key) == int:
                return (lambda x: x.session == key)
            elif type(key) == slice:
                return (lambda x: int(x.session) <= key[1] and int(x.session) > key[0])
            else:
                raise AttributeError('unsupported data type for session slicing')

        def gen_point_filter(key):
            return None


        if type(key) != tuple:
            key = (key, slice(None, None, None), slice(None, None, None))
        while(len(key) < 3):
            key += (slice(None, None, None), )

        id_filter = gen_id_filter(key[0])
        sess_filter = gen_sess_filter(key[1])
        point_filter = gen_point_filter(key[2])
        return list(filter(point_filter, filter(sess_filter, filter(id_filter, self.points))))

    def copy(self, id_filter=None, sess_filter=None, point_filter=None, newname=None, softlinked=False):
        if newname is None:
            newname = str(uuid.uuid4())
        set = type(self)({'dir': self.dir, 'name': newname}, self.seed, self.context)

        self.log.info("Creating new dataset " + newname + ", copy of " + self.name)

        for point in filter(point_filter, filter(sess_filter, filter(id_filter, self.points))):
            point.copy(set, softlink=softlinked)

        set.meta = {"name": newname, "original": self.name, "datatype": self.meta["datatype"]}
        if "original_meta" not in self.meta:
            set.meta["original_meta"] = self.meta
        else:
            set.meta["original_meta"] = self.meta["original_meta"]
        if softlinked:
            set.meta["softlinked"] = True
        with open(os.path.join(self.dir, newname + ".meta.yaml"), "w") as file:
            file.write("---\n" + yaml.dump(set.meta))

        return set.run([])

    def merge_and_rename(self, datasets, mappings, newname=None, softlinked=False):
        # sanity checks
        if not len(set(map(lambda x: type(x), [self] + datasets))) == 1:
            self.log.error("Cannot merge dataests with different types!")
        if not len(datasets) + 1 == len(mappings):
            self.log.error("Merging: Mapping has invalid length!")
        a = [x for mapping in mappings for x in mapping.values()]
        if not len(a) == len(list(set(a))):
            self.log.error("Merging: Destination point names are not unique!")

        # Setting up new dataset
        if newname is None:
            newname = str(uuid.uuid4())
        newset = type(self)({'dir': self.dir, 'name': newname}, self.seed, self.context)
        self.log.info("Creating new dataset {}, merge of {}".format(newname, ", ".join(map(lambda x: x.name, [self] + datasets))))

        # copy the points
        for ds, mapping in zip([self] + datasets, mappings):
            for point in ds[:]:
                id = "{}.{}.{}".format(point.identity, point.session, point.pointname)
                if id in mapping:
                    point.copy(newset, softlink=softlinked, newname=mapping[id])

        # meta info stuff
        newset.meta = {"name": newname, "original": "|".join(map(lambda x: x.name, [self] + datasets)), "datatype": self.meta["datatype"]}
        if "original_meta" not in self.meta:
            newset.meta["original_meta"] = self.meta
        else:
            newset.meta["original_meta"] = self.meta["original_meta"]
        if softlinked:
            newset.meta["softlinked"] = True
        with open(os.path.join(self.dir, newname + ".meta.yaml"), "w") as file:
            file.write("---\n" + yaml.dump(newset.meta))
        return newset.run([])


    def save_meta(self):
        with open(os.path.join(self.dir, self.name + ".meta.yaml"), "w") as file:
            file.write("---\n" + yaml.dump(self.meta))

    def delete(self):
        pass

class AbstractDataPoint:
    def __init__(self):
        self.attr = {}

    def __getattr__(self, atr):
        self.load_attrs()
        if atr not in self.attr:
            return None
        else:
            return self.attr[atr]

    def save_attr(self, key, value):
        self.load_attrs()
        self.attr[key] = value
        self.save_attrs()

    def save_attr_batch(self, attributes):
        self.load_attrs()
        for k, v in attributes.items():
            self.attr[k] = v
        self.save_attrs()

    def load_attrs(self):
        pass

    def save_attrs(self):
        pass

    def copy(self, newset, softlink=False, newname=None):
        pass

    def replace(self, new, softlink=False):
        pass

    def update_fn(self):
        pass

    def delete_file(self):
        pass
