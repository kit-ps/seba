from .abstract import AbstractProcessing

import logging
import random

class NormsessionsProcessing(AbstractProcessing):
    name = "normsessions"
    softlinked = True

    def process_all(self):
        while self.reorder():
            pass

    def reorder(self):
        changes = False
        for identity in self.dataset.identities:
            points = self.dataset[identity]
            id_sessions = sorted(list(set(map(lambda x: x.session, points))))
            id_s_index = 0
            for dsess in self.dataset.sessions:
                if id_s_index == len(id_sessions):
                    break
                if dsess not in id_sessions:
                    for point in list(filter(lambda x: x.session == id_sessions[id_s_index], points)):
                        point.session = dsess
                        changes = True
                        point.update_fn()
                id_s_index += 1
        return changes
