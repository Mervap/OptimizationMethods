import math

import numpy as np


def markers(min_v: float, max_v: float):
    diff = (max_v - min_v) / 7
    return [min_v + diff * (i + 1) * (1.15 ** i) for i in range(7)]


def plot_graph(ind, fig, opts, sts, mns, title, per, x_expand, y_expand, plt_size=(6, 2), labels=None):
    if labels is None:
        labels = sts
    f = opts[0].f
    logs = list(map(lambda x: x.log_frame(), opts))
    logs = list(map(lambda x: x.iloc[-max(3, int(len(x) * per / 100)):], logs))

    x_min = min(list(map(lambda x: min(x['x']), logs)))
    x_max = max(list(map(lambda x: max(x['x']), logs)))
    x_diff = x_max - x_min
    y_min = min(list(map(lambda x: min(x['y']), logs)))
    y_max = max(list(map(lambda x: max(x['y']), logs)))
    y_diff = y_max - y_min

    ax = fig.add_subplot(plt_size[0], plt_size[1], ind)
    n_cnt = 300
    x_axis_min = x_min - x_diff / 2 - x_expand / 2
    x_axis_max = x_max + x_diff / 2 + x_expand / 2
    y_axis_min = y_min - y_diff / 2 - y_expand / 2
    y_axis_max = y_max + y_diff / 2 + y_expand / 2
    ax.set_xlim([x_axis_min, x_axis_max])
    ax.set_ylim([y_axis_min, y_axis_max])
    ax.set_xlabel('X', fontsize=14)
    ax.set_ylabel('Y', fontsize=14)
    ax.set_title(title, fontsize=14)

    xs, ys = np.meshgrid(np.linspace(x_axis_min, x_axis_max, n_cnt), np.linspace(y_axis_min, y_axis_max, n_cnt))

    f_values = np.vectorize(lambda x, y: f(x, y))(xs, ys)
    f_min = min(list(map(lambda x: min(x['f']), logs)))
    f_max = max(list(map(lambda x: max(x['f']), logs)))

    grid_min = np.min(f_values)
    grid_max = np.max(f_values)

    min_v = math.sqrt(f_min * grid_min)
    if f_min < 0 or grid_min < 0:
        min_v = -min_v
    max_v = math.sqrt(f_max * grid_max)
    if f_max < 0 and grid_max < 0:
        max_v = -max_v
    qx = ax.contour(xs, ys, f_values, markers(min_v, max_v))

    lg = math.log10(max_v - min_v)
    if lg < 0:
        pw = int(abs(lg)) + 1
    else:
        if lg > 2:
            pw = 0
        else:
            pw = 1
    ax.clabel(qx, fontsize=8, fmt=f'%.{pw}f', inline=1)
    ax.pcolormesh(xs, ys, f_values, cmap='YlOrBr_r', shading='nearest')

    arrow_color = ["gold", "green", "gray", "blue"]

    for i in range(len(mns)):
        st = sts[i]
        ax.plot(st[0], st[1], 'o', color=arrow_color[i], ms=5, mew=3, label=labels[i])
        mn = mns[i]
        ax.plot(mn[0], mn[1], 'x', color=arrow_color[i], ms=8 + (len(mns) - i) * 3, mew=3)

    for i in range(len(opts)):
        opt = opts[i]
        clr = arrow_color[i]
        last = None
        for ind, e in opt.log_frame().iterrows():
            if last is not None:
                dx = e['x'] - last[0]
                dy = e['y'] - last[1]
                ar_len = math.sqrt(dx ** 2 + dy ** 2)
                head_len = min(2, ar_len / 7)
                width = math.sqrt((x_max - x_min) * (y_max - y_min)) / 250
                ax.arrow(last[0], last[1], e['x'] - last[0], e['y'] - last[1], width=width, color=clr,
                         head_length=head_len, length_includes_head=True)
            last = (e['x'], e['y'])
    ax.legend()
