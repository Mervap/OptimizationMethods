from math import exp

import numpy as np

from lab1.task1.golden_ratio import GoldenRatio
from lab1.watcher import Watcher
from lab2.newton import Newton
from lab2.conjgrad import ConjGrad


functions = [lambda x, y: 100 * (y - x) ** 2 + (1 - x) ** 2,
             lambda x, y: 100 * (y - x ** 2) ** 2 + (1 - x) ** 2,
             lambda x, y: (-1) * (2 * exp(-((x - 1) / 2) ** 2 - (y - 1) ** 2) +
                                  3 * exp(-((x - 2) / 3) ** 2 - ((y - 3) / 2) ** 2))]
gradients = [lambda x, y: np.array([-200 * (y - x) + 2 * x - 2, 200 * (y - x)]),
             lambda x, y: np.array([-400 * (y - x ** 2) * x - 2 * (1 - x), 200 * (y - x ** 2)]),
             lambda x, y: (-1.0) * np.array([2 * exp(-((x - 1) / 2) ** 2 - (y - 1) ** 2) * (-(x - 1) / 2) +
                                             3 * exp(-((x - 2) / 3) ** 2 - ((y - 3) / 2) ** 2) * (2.0 / 3.0) * (
                                                     -(x - 2) / 3),
                                             2 * exp(-((x - 1) / 2) ** 2 - (y - 1) ** 2) * (-2 * (y - 1)) +
                                             3 * exp(-((x - 2) / 3) ** 2 - ((y - 3) / 2) ** 2) * (-((y - 3) / 2))])]
hesse = [lambda x, y: np.array([[202, -200], [-200, 200]]),
         lambda x, y: np.array([[-400 * ((y - x ** 2) - 2 * x ** 2) + 2, -400 * x], [-400 * x, 200]]),
         lambda x, y: (-1) * np.array(
             [[3 * ((4 / 81) * (x - 2) ** 2 * exp((-1 / 9) * (x - 2) ** 2 - (1 / 4) * (y - 3) ** 2) -
                    (2 / 9) * exp((-1 / 9) * (x - 2) ** 2 - (1 / 4) * (y - 3) ** 2)) +
               2 * ((1 / 4) * (x - 1) ** 2 * exp((-(1 / 4) * (1 - x) ** 2 - (y - 1) ** 2)) -
                    (1 / 2) * exp(-(1 / 4) * (1 - x) ** 2 - (y - 1) ** 2)),
               (1 / 3) * (x - 2) * (y - 3) * exp(-(1 / 9) * (x - 2) ** 2 - (1 / 4) * (y - 3) ** 2) +
               2 * (x - 1) * (y - 1) * exp(-(1 / 4) * (x - 1) ** 2 - (y - 1) ** 2)],
              [(1 / 3) * (x - 2) * (y - 3) * exp(-(1 / 9) * (x - 2) ** 2 - (1 / 4) * (y - 3) ** 2) +
               2 * (x - 1) * (y - 1) * exp(-(1 / 4) * (x - 1) ** 2 - (y - 1) ** 2),
               3 * ((1 / 4) * (y - 3) ** 2 * exp((-1 / 9) * (x - 2) ** 2 - (1 / 4) * (3 - y) ** 2) -
                    (1 / 2) * exp((-1 / 9) * (x - 2) ** 2 - (1 / 4) * (3 - y) ** 2)) +
               2 * (4 * (y - 1) ** 2 * exp((-1 / 4) * (x - 1) ** 2 - (y - 1) ** 2) -
                    2 * exp((-1 / 4) * (x - 1) ** 2 - (y - 1) ** 2))]])]

initial_points = [
    np.array([0, 1]),
    np.array([2, 2]),
    np.array([-1.0, 1.0]),
    np.array([3.0, 2.0]),
    np.array([0.1, -0.5]),
]

def simple_test(alg):
    for j in range(len(functions)):
        f = Watcher(functions[j], gradients[j], hesse[j])
        print(j)
        for ip in initial_points:
            opt = alg(f, 1e-5, GoldenRatio)
            try:
                mn = opt.opt(ip)
                print(mn, opt.get_iterations_cnt())
            except RuntimeError as e:
                if hasattr(e, 'message'):
                    print(e.message)
                else:
                    print(e)

simple_test(Newton)
simple_test(ConjGrad)