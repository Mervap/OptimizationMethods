import math

import numpy as np
from lab1.task1.golden_ratio import GoldenRatio
from lab1.task1.fibonacci import Fibonacci
from lab1.task1.dichotomy import Dichotomy
from lab1.watcher import Watcher
from lab1.task2.gradient_descent import GradientDescent
import matplotlib.pyplot as plt

def generate_level_markers(minvalue: float, maxvalue: float):
    amount = 10
    level_separator = (maxvalue - minvalue * 0.001) / float(amount)
    level_lines = [minvalue * 0.001 + level_separator * (1.1 ** i) * float(i) for i in range(amount)]
    if maxvalue < 0:
        level_separator = (maxvalue * 0.5 - minvalue) / float(amount)
        level_lines = [minvalue + level_separator * float(i) for i in range(amount)]
        level_separator *= 2
    return level_lines

def plot_graph(ind, fig, opt, title, per):
    f = opt.f
    log = opt.log_frame()
    log = log.iloc[-max(3, int(len(log) * per / 100)):]

    x_min = min(log['x'])
    x_max = max(log['x'])
    x_diff = x_max - x_min
    y_min = min(log['y'])
    y_max = max(log['y'])
    y_diff = y_max - y_min

    ax = fig.add_subplot(2, 2, ind)
    n_cnt = 300
    ax.set_xlim([x_min - x_diff / 2, x_max + x_diff / 2])
    ax.set_ylim([y_min - y_diff / 2, y_max + y_diff / 2])
    ax.set_xlabel('X', fontsize=14)
    ax.set_ylabel('Y', fontsize=14)
    ax.set_title(title, fontsize=14)

    xs, ys = np.meshgrid(
        np.linspace(x_min - x_diff / 2, x_max + x_diff / 2, n_cnt),
        np.linspace(y_min - y_diff / 2, y_max + y_diff / 2, n_cnt)
    )

    v_func = np.vectorize(lambda x, y: f(x, y))
    colors = v_func(xs, ys)

    f_min = min(log['f'])
    f_max = max(log['f'])
    grid_min = np.min(colors)
    grid_max = np.max(colors)

    min_v = math.sqrt(f_min * grid_min)
    if f_min < 0 or grid_min < 0:
        min_v = -min_v
    max_v = math.sqrt(f_max * grid_max)
    if f_max < 0 and grid_max < 0:
        max_v = -max_v
    qx = ax.contour(xs, ys, colors, generate_level_markers(min_v, max_v))
    ax.clabel(qx, fontsize=8, fmt='%.5f', inline=1)
    ax.pcolormesh(xs, ys, colors, cmap='YlOrBr_r')

    ax.plot(mn[0], mn[1], 'x', color='red', ms=10, mew=3)

    last = None
    for ind, e in opt.log_frame().iterrows():
        if last is not None:
            dx = e['x'] - last[0]
            dy = e['y'] - last[1]
            ar_len = math.sqrt(dx ** 2 + dy ** 2)
            head_len = min(2, ar_len / 7)
            width = (x_max - x_min) / 300
            ax.arrow(last[0], last[1], e['x'] - last[0], e['y'] - last[1], width=width, alpha=0.7, color='gray', head_length=head_len, length_includes_head=True)
        last = (e['x'], e['y'])

task2_fun = [
    (lambda x, y: 3 * x ** 2 + 7 * y ** 2 + y * x - x, lambda x, y: np.array([6 * x + y - 1, x + 14 * y])),
    # (lambda x, y: 100 * (y - x) ** 2 + (1 - x) ** 2, lambda x, y: np.array([-200 * (y - x) + 2 * x - 2, 200 * (y - x)])),
    # (lambda x, y: 100 * (y - x ** 2) ** 2 + (1 - x) ** 2, lambda x, y: np.array([-400 * (y - x ** 2) * x - 2 * (1 - x), 200 * (y - x ** 2)]))
    (lambda x, y: (14 - x) ** 2 + 88 * (y - 4 * x + 7) ** 2, lambda x, y: np.array([2 * (1409 * x - 352 * y - 2478), 176 * (7 - 4 * x + y)])),
]

f1_str = '$f(x, y) = 3x^2+7y^2+2yx-x$'
f2_str = '$f(x, y) = (14-x)^2+88(y - 4x + 7)^2$'

params = [
    [(f1_str, 100), (f1_str + ' (last iters)', 10)],
    [(f2_str, 100), (f2_str + ' (last iters)', 5)],
    # [(f2_str, -2, 5, -1, 4)],
    # [(f2_str, -7, 7, -5, 45)],
]

fig = plt.figure(figsize=(16, 14))
i = 0
for j in range(len(task2_fun)):
    test = task2_fun[j]
    f = Watcher(test[0], test[1])
    opt = GradientDescent(f, 1e-5, GoldenRatio)
    # mn = opt.opt([13, 46])
    mn = opt.opt([0, 1])
    for (title, per) in params[j]:
        i = i + 1
        plot_graph(i, fig, opt, title, per)
fig.tight_layout()