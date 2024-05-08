import os
import os.path
import yaml


class Identity:
    def __init__(self, setpath, name):
        self.setpath = setpath
        self.name = name
        self.npoints = 0
        self.attr = False

    def __getattr__(self, atr):
        self.load_attr()
        if atr not in self.attr:
            return None
        else:
            return self.attr[atr]

    def load_attr(self):
        if self.attr is False:
            # lazy loading
            if os.path.exists(os.path.join(self.setpath, self.name + ".yaml")):
                with open(os.path.join(self.setpath, self.name + ".yaml"), "r") as file:
                    self.attr = yaml.load(file, Loader=yaml.SafeLoader)
            else:
                self.attr = {}

    def save_attr(self, key, value):
        self.load_attr()
        self.attr[key] = value
        self.save_attr_file()

    def save_attr_batch(self, attributes):
        self.load_attr()
        for k, v in attributes.items():
            self.attr[k] = v
        self.save_attr_file()

    def save_attr_file(self):
        with open(os.path.join(self.setpath, self.name + ".yaml"), "w") as file:
            file.write("---\n" + yaml.dump(self.attr))
