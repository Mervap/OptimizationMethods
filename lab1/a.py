import math

import numpy as np
from lab1.task1.golden_ratio import GoldenRatio
from lab1.task1.fibonacci import Fibonacci
from lab1.task1.dichotomy import Dichotomy
from lab1.task2.binary_search import BinarySearch
from lab1.task2.fastest_gradient_descent import FastestGradientDescent
from lab1.watcher import Watcher
import matplotlib.pyplot as plt

def markers(min_v: float, max_v: float):
    diff = (max_v - min_v) / 7
    return [min_v + diff * (i + 1) * (1.15 ** i) for i in range(7)]


def plot_graph(ind, fig, opts, mns, title, per, x_expand, y_expand):
    f = opts[0].f
    logs = list(map(lambda x: x.log_frame(), opts))
    logs = list(map(lambda x: x.iloc[-max(3, int(len(x) * per / 100)):], logs))

    x_min = min(list(map(lambda x: min(x['x']), logs)))
    x_max = max(list(map(lambda x: max(x['x']), logs)))
    x_diff = x_max - x_min
    y_min = min(list(map(lambda x: min(x['y']), logs)))
    y_max = max(list(map(lambda x: max(x['y']), logs)))
    y_diff = y_max - y_min

    ax = fig.add_subplot(6, 2, ind)
    n_cnt = 300
    ax.set_xlim([x_min - x_diff / 2 - x_expand / 2, x_max + x_diff / 2 + x_expand / 2])
    ax.set_ylim([y_min - y_diff / 2 - y_expand / 2, y_max + y_diff / 2 + y_expand / 2])
    ax.set_xlabel('X', fontsize=14)
    ax.set_ylabel('Y', fontsize=14)
    ax.set_title(title, fontsize=14)

    xs, ys = np.meshgrid(
        np.linspace(x_min - x_diff / 2 - x_expand / 2, x_max + x_diff / 2 + x_expand / 2, n_cnt),
        np.linspace(y_min - y_diff / 2 - y_expand / 2, y_max + y_diff / 2 + y_expand / 2, n_cnt)
    )

    v_func = np.vectorize(lambda x, y: f(x, y))
    colors = v_func(xs, ys)

    f_min = min(list(map(lambda x: min(x['f']), logs)))
    f_max = max(list(map(lambda x: max(x['f']), logs)))
    grid_min = np.min(colors)
    grid_max = np.max(colors)

    min_v = math.sqrt(f_min * grid_min)
    if f_min < 0 or grid_min < 0:
        min_v = -min_v
    max_v = math.sqrt(f_max * grid_max)
    if f_max < 0 and grid_max < 0:
        max_v = -max_v
    qx = ax.contour(xs, ys, colors, markers(min_v, max_v))

    lg = math.log10(max_v - min_v)
    if lg < 0:
        pw = int(abs(lg)) + 1
    else:
        if lg > 2:
            pw = 0
        else:
            pw = 1
    ax.clabel(qx, fontsize=8, fmt=f'%.{pw}f', inline=1)
    ax.pcolormesh(xs, ys, colors, cmap='YlOrBr_r', shading='nearest')

    arrow_color = ["gold", "green", "gray", "blue"]

    for i in range(len(mns)):
        mn = mns[i]
        ax.plot(mn[0], mn[1], 'x', color=arrow_color[i], ms=10, mew=3)

    last = None
    for i in range(len(opts)):
        opt = opts[i]
        clr = arrow_color[i]
        for ind, e in opt.log_frame().iterrows():
            if last is not None:
                dx = e['x'] - last[0]
                dy = e['y'] - last[1]
                ar_len = math.sqrt(dx ** 2 + dy ** 2)
                head_len = min(2, ar_len / 7)
                width = (x_max - x_min) / 250
                ax.arrow(last[0], last[1], e['x'] - last[0], e['y'] - last[1], width=width, color=clr,
                         head_length=head_len, length_includes_head=True)
            last = (e['x'], e['y'])


task2_fun = [
    # (lambda x, y: 3 * x ** 2 + 7 * y ** 2 + y * x - x, lambda x, y: np.array([6 * x + y - 1, x + 14 * y])),
    (lambda x, y: (14 - x) ** 2 + 88 * (y - 4 * x + 7) ** 2,
     lambda x, y: np.array([2 * (1409 * x - 352 * y - 2478), 176 * (7 - 4 * x + y)])),
    (lambda x, y: x * math.exp(-(x ** 2 + y ** 2)),
     lambda x, y: np.array([(1 - 2 * x ** 2) * math.exp(-(x ** 2 + y ** 2)),
                            -2 * x * y * math.exp(-(x ** 2 + y ** 2))])),
]

f1_str = '$f(x, y) = 3x^2+7y^2+2yx-x$'
f2_str = '$f(x, y) = (14-x)^2+88(y - 4x + 7)^2$'
f3_str = '$f(x, y) = xe^{-x^2-y^2}$'

params = [
    # [(f1_str, 100, 0, 0), (f1_str + ' (last iters)', 10, 0, 0)],
    [(f2_str, 100, 0, 0), (f2_str + ' (last iters)', 5, 0, 0)],
    [(f3_str, 100, 2, 2)],
]

sts = [
    [0.5, 0.1],
    [0, 1],
    [-3, 0.4],
    [-1, 1]
]


frame = []
opts = []
mns = []
for j in range(len(task2_fun)):
    test = task2_fun[j]
    f = Watcher(test[0], test[1])
    _opts = []
    _mns = []
    for st in sts:
        for opt_t in [GoldenRatio, Fibonacci, Dichotomy, BinarySearch]:
            print(j, st, opt_t.name)
            opt = FastestGradientDescent(f, 1e-5, opt_t)
            try:
                mn = opt.opt(st)
                frame.append([j, opt_t.name, st, mn, opt.get_iterations_cnt(), f.invocations, f.grad_invocations])
                if opt_t == GoldenRatio:
                    _mns.append(mn)
                    _opts.append(opt)
            except Exception as error:
                frame.append([j, opt_t.name, repr(error), None, None, None])
    opts.append(_opts)
    mns.append(_mns)