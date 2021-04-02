import numpy as np
from lab1.task1.golden_ratio import GoldenRatio
from lab1.task1.fibonacci import Fibonacci
from lab1.task1.dichotomy import Dichotomy
from lab1.watcher import Watcher
from lab1.task2.gradient_descent import GradientDescent
import matplotlib.pyplot as plt

task2_fun = [
    (lambda x, y: 3 * x ** 2 + 7 * y ** 2 + y * x - x, lambda x, y: np.array([6 * x + y - 1, x + 14 * y])),
    (lambda x, y: (14 - x) ** 2 + 88 * (y - 4 * x + 7) ** 2, lambda x, y: np.array([2 * (1409 * x - 352 * y - 2478), 176 * (7 - 4 * x + y)]))
]

# for test in tests:
#     f = Watcher(test[0], test[1])
#     for method in [Dichotomy, GoldenRatio, Fibonacci]:
#         opt = GradientDescent(f, 1e-5, method)
#         res = opt.opt([0, 1])
#         print(res, end=" ")
#         print(f.invocations, end=" ")
#         print(f.grad_invocations)
#     print()
test = task2_fun[1]
f = Watcher(test[0], test[1])
opt = GradientDescent(f, 1e-5, GoldenRatio)
opt.opt([13, 46])
print(f.invocations)

fig = plt.figure(figsize=(16, 14))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('X', fontsize=15)
ax.set_ylabel('Y', fontsize=15)
fr = opt.log_frame()
print(len(fr.iloc[-1:, ]))

last = None
for ind, e in fr.iloc[100:].iterrows():
    if last is not None:
        ax.arrow(last[0], last[1], e['x'] - last[0], e['y'] - last[1], width=1e-3 / 4)
    last = (e['x'], e['y'])