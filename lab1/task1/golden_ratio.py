import math

from lab1.task1.reused_opt import ReusedOpt


class GoldenRatio(ReusedOpt):
    golden_c = (math.sqrt(5) - 1) / 2
    name = "Golden ratio"

    def _step(self, x1, x2):
        diff = x2 - x1
        return x2 - GoldenRatio.golden_c * diff, x1 + GoldenRatio.golden_c * diff
