from ..abstract import AbstractAnonymization


class AbstractMotionAnonymization(AbstractAnonymization):
    name = "abstractmotion"

    def anonymize_all(self):
        for point in self.dataset.datapoints.values():
            data = point.load()
            anon_data = self.anonymize(point, data)
            point.save(anon_data)

    def anonymize(self, point, data):
        return None
