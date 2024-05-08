class AbstractPrivacy:
    def get_encoding(self, point):
        """Return an encoding vector for a datapoint.
        Method expects that self.train() has been called beforehand.
        """
        raise NotImplementedError()
