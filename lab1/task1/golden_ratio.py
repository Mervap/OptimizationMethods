import math

from lab1.task1.reused_opt import ReusedOpt


class GoldenRatio(ReusedOpt):
    golder_c = (math.sqrt(5) - 1) / 2

    def _step(self, x1, x2):
        diff = x2 - x1
        return x2 - GoldenRatio.golder_c * diff, x1 + GoldenRatio.golder_c * diff
