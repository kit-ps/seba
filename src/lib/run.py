from .utils import recursive_replace
from .datamanager import DatasetManager

import copy
import importlib
import logging
import os


class Run:
    def __init__(self, config, round=0):
        self.config = recursive_replace(config, "$ROUND$", round)
        self.orig_config = copy.deepcopy(self.config)
        self.seed = self.config['seed'] if 'seed' in self.config else 'seba'
        self.datatype = 'abstract'
        self.datadir = os.path.join(os.getcwd(), "data")
        self.sets = {}
        self.log = logging.getLogger("seba.run")
        self.log.info("=" * 30)
        self.log.info("Starting new run.\n\tConfiguration: " + str(config))

    def run(self):
        for step in self.config['steps']:
            context = {'config': self.config, 'datasets': self.sets}
            self.run_step(step, context)

    def run_step(self, conf, context):
        step = self.step_conf(conf)
        self.log.info(">> Next Step is {}: {}".format(step['type'], step['name']))
        in_data = self.get_data(step['in'])
        module_fn = self.get_module(step['type'], step['name'])

        if not self.find_cached(step, in_data, module_fn.nout, context):
            module = module_fn(step['params'], step['seed'], context)
            out_data = module.run(in_data)
            self.handle_data(step['out'], out_data)
            self.log.info(">> Step complete.")

    def step_conf(self, step):
        default = {
            'type': None,
            'name': None,
            'params': {},
            'seed': self.seed,
            'in': [],
            'out': []
        }
        return default | step

    def get_module(self, type, name):
        class_name = name.capitalize() + type.capitalize()
        try:
            mod = importlib.import_module("src.{}.{}".format(type, name))
        except ModuleNotFoundError:
            mod = importlib.import_module("src.{}.{}.{}".format(type, self.datatype, name))
        abstr = importlib.import_module("src.{}.abstract".format(type))
        cls = getattr(mod, class_name)
        if not issubclass(cls, getattr(abstr, 'Abstract' + type.capitalize())):
            raise AttributeError("{} does not inherit from abstract.".format(class_name))
        return cls

    def get_data(self, ins):
        if type(ins) is not list:
            ins = [ins]
        return list(map(
            lambda x: self.sets[x] if x in self.sets else x,
            ins
        ))

    def handle_data(self, names, outs):
        if type(names) is not list:
            names = [names]
        if type(outs) is not list:
            outs = [outs]
        for n, o in zip(names, outs):
            try:
                self.datatype = o.meta['datatype']
                self.datadir = o.dir
            except:
                pass
            self.sets[n] = o

    def find_cached(self, conf, in_data, nout, context):
        if nout == 0:
            return False
        outs = []
        params = {
            "original": "|".join(map(lambda x: x.name, in_data)),
            "type": conf['type'],
            "name": conf['name'],
            "seed": conf['seed'],
            "params": {key: val for key, val in conf["params"].items() if key != "opt"}
        }
        for i in range(nout):
            params['part'] = i
            outs.append(DatasetManager.get_matching(params, self.datadir, self.get_module('dataset', self.datatype), self.seed, context))

        if None in outs:
            return False
        else:
            self.handle_data(conf['out'], outs)
            self.log.info(">> Step skipped. Using cached sets {}.".format(", ".join(map(lambda x: x.name, outs))))
            return True
