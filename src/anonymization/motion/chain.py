from .abstract import AbstractMotionAnonymization
from ...lib.module_loader import ModuleLoader


class ChainAnonymization(AbstractMotionAnonymization):
    """Apply multiple anonymizations in a row

    Required pips:
        none
    Parameters:
        - anonymizations (list[dict]): list of anonymizations to apply in a row
            For each anonymization, a dictionary with {'name': 'anon_name', 'params': { PARAMETERS }} is expected.
    """

    name = "chain"

    def validate_config(self):
        if "anonymizations" not in self.config:
            self.log.error("Missing anonymizations")

    def anonymize(self, mocap, data):
        for anon in self.config["anonymizations"]:
            a = ModuleLoader.get_anonymization_by_name(anon["name"], "gait")(anon["params"], self.dataset)
            data = a.anonymize(mocap, data)
        return data
