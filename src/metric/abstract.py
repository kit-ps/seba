import logging


class AbstractMetric:
    def __init__(self, resultset):
        self.result = resultset
        self.log = logging.getLogger("seba.metric")

    def run(self):
        me = self.calc()
        self.print_debug(me)
        self.print_csv(me)
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
