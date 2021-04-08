import numpy as np

"""
f(x, y) = 100(y - x)^2 + (1 - x)^2
f(x, y) = 101x^2 − 200xy + 100y^2 − 2x + 1
f(x, y) has a minimum in the same point as f(x, y) = 101x^2 − 200xy + 100y^2 − 2x
f(x, y) = (101x^2 - 200xy + 100y^2) - 2x

Q(x)/2 =  (101 -100)
        (-100 100)
and
b = (-2)
    (0)
"""
mA = np.array([[202, -200], [-200, 200]])
mB = np.array([2, 0])
mX = np.array([0, 0])

"""
f(x, y) = 100(y−x^2)^2 + (1 − x)^2
f(x, y) = 100 x^4 - 200 x^2 y + x^2 - 2 x + 100 y^2 + 1
f(x, y) has a minimum in the same point as f(x, y) = 100 x^4 - 200 x^2 y + x^2 - 2 x + 100 y^2
f(x, y) = (100 x^4 - 200 x^2 y + 100 y^2) + -2x + x^2
??? it is not quadratic

"""


def conjGradient(Q, b, x, eps):
    r = b - np.dot(Q, x)
    u = r
    rrT_prev = np.dot(np.transpose(r), r)
    for _ in range(len(b)):
        Qu = np.dot(Q, u)
        alpha = rrT_prev / np.dot(np.transpose(u), Qu)
        x = x + np.dot(alpha, u)
        r = r - np.dot(alpha, Qu)
        rrT = np.dot(np.transpose(r), r)
        if np.sqrt(rrT) < eps:
            break
        u = r + (rrT / rrT_prev) * u
        rrT_prev = rrT
    return x


#solves Ax = b. do not confuse the sign of b.
t = conjGradient(mA, mB, mX, 1e-8)

print(np.asarray(t))