from .abstract import AbstractSplitter

import random

class SinglerandomsessionSplitter(AbstractSplitter):
    """Simple splitter to select a single session - for ids with multiple sessions, select random one

    Required pips:
        none

    Parameters:
        (string) target_session: the name of the single session to select
    """

    name = "singlerandomsession"
    random = True
    nin = 1
    nout = 1

    def split(self, in_sets):
        keep_id_sess = []

        for identity in sorted(in_sets[0].identities):
            id_points = in_sets[0][identity]
            id_sessions = sorted(list(set(map(lambda x: x.session, id_points))))
            choice = random.choice(id_sessions)
            keep_id_sess.append((identity, choice))

        return [in_sets[0].copy(
            point_filter=(lambda x: (x.identity, x.session) in keep_id_sess),
            softlinked=True
        )]
