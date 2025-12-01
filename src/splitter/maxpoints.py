from .abstract import AbstractSplitter

class MaxpointsSplitter(AbstractSplitter):
    """Use random identities in the set

    Required pips:
        none

    Parameters:
        - (int) points: number of points per label to select
    """

    name = "maxpoints"
    random = False
    nin = 1
    nout = 1

    def validate_config(self):
        if "points" not in self.config:
            raise AttributeError("Splitter: config: Missing number of points")

    def split(self, in_sets):
        keep_points = []
        for label in in_sets[0].labels:
            if "from" in self.config and self.config['from'] == 'back':
                keep_points += in_sets[0][label][(-1) * self.config['points']:]
            else:
                keep_points += in_sets[0][label][: self.config['points']]

        point_filter = (lambda x: x in keep_points)

        return [in_sets[0].copy(point_filter=point_filter, softlinked=True)]
