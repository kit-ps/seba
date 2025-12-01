import logging
import yaml

class AbstractMetric:
    nout = 0

    def __init__(self, config, seed, context):
        self.config = config
        self.context = context
        self.log = logging.getLogger("seba.metric")

    def run(self, resultset):
        self.result = resultset[0]
        me = self.calc()
        self.print_debug(me)
        self.print_csv(me)
        self.save_result(me)
        return me

    def calc(self):
        return {}

    def print_csv(self, me):
        print(",".join(me.keys()))
        print(",".join([str(x) for x in me.values()]))

    def print_debug(self, me):
        self.log.debug("Results ID = {}".format(self.result.id))
        for k, v in me.items():
            self.log.debug("{} = {}".format(k, v))

    def save_result(self, me):
        a = {
            self.result.id: {
                'config': self.context['config'],
                'datasets': dict(map(lambda x: (x[0], x[1].name), filter(lambda x: hasattr(x[1], 'name'), self.context['datasets'].items()))),
                'result': me
            }
        }
        with open('results.yaml', 'a') as f:
            f.write(yaml.dump(a))
