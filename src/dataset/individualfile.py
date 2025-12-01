from .abstract import AbstractDataset, AbstractDataPoint

import os
import shutil
import yaml


class FileDataPoint(AbstractDataPoint):
    def __init__(self, setpath, setmeta, filename):
        self.setpath = setpath
        self.setmetadata = setmeta
        self.filename = filename
        self.attr = False

        if len(filename.split('.')) == 4:
            self.identity, self.session, self.pointname, self.ext = filename.split('.')
        else:
            self.identity, self.pointname, self.ext = filename.split('.')
            self.session = 0
        self.label = self.identity

    def load_attrs(self):
        if self.attr is False:
            # lazy loading
            if os.path.exists(os.path.join(self.setpath, self.filename.replace('.' + self.ext, '.yaml'))):
                with open(os.path.join(self.setpath, self.filename.replace('.' + self.ext, '.yaml')), "r") as file:
                    self.attr = yaml.load(file, Loader=yaml.SafeLoader)
            else:
                self.attr = {}

    def save_attrs(self):
        with open(os.path.join(self.setpath, self.filename.replace('.' + self.ext, '.yaml')), "w") as file:
            file.write("---\n" + yaml.dump(self.attr))

    def get_path(self):
        return os.path.join(self.setpath, self.get_filename())

    def get_filename(self):
        return self.filename

    def update_fn(self):
        fn = "{}.{}.{}.{}".format(self.identity, self.session, self.pointname, self.ext)
        if fn != self.filename:
            oldpath = self.get_path()
            self.filename = fn
            os.rename(oldpath, self.get_path())

    def delete_file(self):
        os.unlink(self.get_path())

    def replace(self, filename, softlink=False):
        fns = [(filename, self.get_path())]
        if os.path.exists(self.get_path().replace('.' + self.ext, '.yaml')):
            fns.append((filename.replace("." + self.ext, '.yaml'), self.get_path().replace('.' + self.ext, '.yaml')))

        for new, old in fns:
            try:
                os.remove(old)
                if softlink:
                    os.symlink(new, old)
                else:
                    shutil.copy(new, old)
            except Exception:
                raise ValueError("Failed to replace file " + old + " with " + new)

    def copy(self, newset, softlink=False, newname=None):
        if newname is not None:
            fn = newname + "." + self.ext
        else:
            fn = "{}.{}.{}.{}".format(self.identity, self.session, self.pointname, self.ext)

        if os.path.exists(os.path.join(self.setpath, self.filename)): # don't copy points where the files have been deleted
            if softlink:
                os.symlink(os.path.join(self.setpath, self.filename), os.path.join(newset.folder, fn))
            else:
                shutil.copy(os.path.join(self.setpath, self.filename), os.path.join(newset.folder, fn))

        metafile_old = self.filename.replace('.' + self.ext, '.yaml')
        metafile_new = fn.replace("." + self.ext, '.yaml')
        if os.path.exists(os.path.join(self.setpath, metafile_old)):
            if softlink:
                os.symlink(os.path.join(self.setpath, metafile_old), os.path.join(newset.folder, metafile_new))
            else:
                shutil.copy(os.path.join(self.setpath, metafile_old), os.path.join(newset.folder, metafile_new))


class IndividualfileDataset(AbstractDataset):
    pointclass = FileDataPoint

    def init(self):
        self.folder = os.path.join(self.dir, self.name)
        os.makedirs(self.folder, exist_ok=True)

    def run(self, datasets):
        self.scan()

        self.log.info(
            "Loaded dataset {} ({} datapoints, {} identities, {} sessions)".format(self.name, len(self.points), len(self.identities), len(self.sessions))
        )
        return self

    def scan(self):
        self.points = []
        files = os.listdir(self.folder)
        files.sort()
        self.log.debug("Checking " + str(len(files)) + " files.")

        for file in files:
            if not file.split('.')[-1] == 'yaml' and 'empty' not in file:
                self.points.append(self.pointclass(self.folder, self.meta, file))

        self.identities = sorted(set(map(lambda x: x.identity, self.points)))
        self.labels = self.identities
        self.sessions = sorted(set(map(lambda x: x.session, self.points)))

    def copy(self, id_filter=None, sess_filter=None, point_filter=None, newname=None, softlinked=False):
        dataset = super().copy(id_filter=id_filter, sess_filter=sess_filter, point_filter=point_filter, newname=newname, softlinked=softlinked)
        for file in list(filter(lambda x: 'empty' in x, os.listdir(self.folder))):
            if softlinked:
                os.symlink(os.path.join(self.folder, file), os.path.join(dataset.folder, file))
            else:
                shutil.copy(os.path.join(self.folder, file), os.path.join(dataset.folder, file))
        return dataset

    def delete(self):
        self.log.warn("Deleting data set " + self.name)
        shutil.rmtree(self.folder)
        os.remove(os.path.join(self.dir, self.name + ".meta.yaml"))
