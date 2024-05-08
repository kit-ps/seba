from .abstract import AbstractMetric

import numpy as np
import scipy.stats as st


class AccuracyMetric(AbstractMetric):
    """Calculate top-k-accuracy, chance-level-top-k-accuracy and normalized-top-k-accuracy

    Required pips:
        - numpy
        - scipy

    Parameters:
        none
    """

    def calc(self):
        export = {}

        # n, hitrate, topNrate
        overall = len(self.result.results)
        success_n = 0
        top3_n = 0
        top5_n = 0
        top10_n = 0
        identities = {}

        self.result.results.sort(key=lambda x: x.identity)

        for result in self.result.to_squashed(strat="min").results:
            if result.identity not in identities:
                identities[result.identity] = []
            identities[result.identity].append(result)

        for result in self.result.to_squashed(strat="min").results:
            if len(result.recognized) > 0:
                if result.is_success():
                    success_n += 1
                    top3_n += 1
                    top5_n += 1
                    top10_n += 1
                elif result.is_topn_success(3):
                    top3_n += 1
                    top5_n += 1
                    top10_n += 1
                elif result.is_topn_success(5):
                    top5_n += 1
                    top10_n += 1
                elif result.is_topn_success(10):
                    top10_n += 1
            else:
                success_n += 1 / len(identities.keys())
                top3_n += min(3 / len(identities.keys()), 1)
                top5_n += min(5 / len(identities.keys()), 1)
                top10_n += min(10 / len(identities.keys()), 1)

        export["ids"] = len(identities.keys())
        export["n"] = len(self.result.results)
        export["hitrate"] = round((success_n / overall), 3)
        export["top1rate"] = round((success_n / overall), 3)
        export["top3rate"] = round((top3_n / overall), 3)
        export["top5rate"] = round((top5_n / overall), 3)
        export["top10rate"] = round((top10_n / overall), 3)

        # chance level rates
        for k in [1, 3, 5, 10]:
            export["cl-top" + str(k) + "rate"] = round(min((k / export["ids"]), 1), 3)
            if 100 - export["cl-top" + str(k) + "rate"] > 0:
                norm_factor = 100 / (100 - export["cl-top" + str(k) + "rate"])
            else:
                norm_factor = 0
            export["top" + str(k) + "rate-normalized"] = round(export["top" + str(k) + "rate"] - export["cl-top" + str(k) + "rate"], 3)
            export["top" + str(k) + "rate-normalized"] = round(export["top" + str(k) + "rate-normalized"] * norm_factor, 3)

        # id-level rates
        accs = []
        for id, results in identities.items():
            hits = 0
            for result in results:
                if result.is_success():
                    hits += 1
            accs.append(hits / len(results))
        if st.sem(accs) == 0.0:
            conf = (np.mean(accs),)
        else:
            conf = st.t.interval(0.95, len(accs) - 1, loc=np.mean(accs), scale=st.sem(accs))
        export["id-mean"] = round(float(np.mean(accs)), 3)
        export["id-conf"] = round(float(np.mean(accs) - conf[0]), 3)

        return export
