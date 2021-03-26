from lab1.task1.dichotomy import Dichotomy
from lab1.task1.golden_ratio import GoldenRatio
from lab1.task1.watcher import Watcher

f = Watcher(lambda x: (x + 4.5) ** 2 - 3)
opt = Dichotomy(f, 1e-5, [-10, 10])
min_v = opt.opt()
print(f'x = {min_v}\ny = {f(min_v)}\n')
print(f.invocations)

opt = GoldenRatio(f, 1e-5, [-10, 10])
min_v = opt.opt()
print(f'x = {min_v}\ny = {f(min_v)}\n')
print(f.invocations)

print(opt.log_frame())
