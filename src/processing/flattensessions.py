from .abstract import AbstractProcessing

import logging
import random

class FlattensessionsProcessing(AbstractProcessing):
    name = "flattensessions"
    softlinked = True

    def process_all(self):
        for identity in self.dataset.identities:
            points = self.dataset[identity]
            points = sorted(points, key=lambda x: '{}.{}'.format(x.session, x.pointname))
            for i in range(len(points)):
                points[i].session = 0
                points[i].pointname = f'{i:03}'
                points[i].update_fn()
