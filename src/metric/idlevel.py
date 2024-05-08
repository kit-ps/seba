from .abstract import AbstractMetric


class IdlevelMetric(AbstractMetric):
    """Identity level accuracy metrics

    Required pips:
        none

    Parameters:
        none
    """

    def calc(self):
        identities = {}

        for result in self.result.to_squashed(strat="min").results:
            if result.identity not in identities:
                identities[result.identity] = []
            identities[result.identity].append(result)

        self.keys = ["identity", "n", "hitrate", "top3rate", "top5rate", "top10rate"]
        self.data = []

        for id, results in identities.items():
            overall = len(results)
            success_n = 0
            top3_n = 0
            top5_n = 0
            top10_n = 0

            for result in results:
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

            hitrate = str(round((success_n / overall) * 100, 2))
            top3rate = str(round((top3_n / overall) * 100, 2))
            top5rate = str(round((top5_n / overall) * 100, 2))
            top10rate = str(round((top10_n / overall) * 100, 2))

            self.data.append([id, str(overall), hitrate, top3rate, top5rate, top10rate])

    def print_csv(self, me):
        print(",".join(self.keys))
        for row in self.data:
            print(",".join(row))

    def print_debug(self, me):
        self.log.debug("no debug log supported")
