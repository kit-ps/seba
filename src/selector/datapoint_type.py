from .abstract import AbstractSelector


class Datapoint_typeSelector(AbstractSelector):
    """Use the entire dataset

    Required pips:
        none

    Parameters:
        (dict) selected_types: A dict of datapoint attributes and the selected values as list, { "attribute1": ["value1", "value2", ...], ... }
    """

    name = "datapoint_type"

    def select(self, set):
        datapoints = set.datapoints

        selected_datapoints = []

        for point in datapoints:
            for attribute in self.config["selected_types"]:
                if getattr(datapoints[point], attribute) in self.config["selected_types"][attribute]:
                    selected_datapoints.append(point)

        return set.copy(only_points=selected_datapoints, softlinked=True)
