#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons


EPSILON = 0.1E-10

STEP = 4

STEP_MIN = 1

STEP_MAX = 50

DOMAIN_START = 0

DOMAIN_END = 1


def domain(nodes_count):
    return np.linspace(DOMAIN_START, DOMAIN_END, nodes_count + 1)


# Исходные данные:
A = 38

# Правая часть диффура
F = lambda x, y, dy: y + A*x*(1-x) + 2*A + 2

# Решение диффура, которое я посчитал ручками (начальные условия дальше)
Y = lambda x: 38*x**2 - 38*x + np.e**(-x) + np.e**x - 2

# Сетка
X = domain(100)

XX = domain(1000)

# Номер последнего узла в сетке (чтобы проверять начальные условия)
N = len(X) - 1


# Начальные условия 
#
# y'(0) = y(0) - A
# y'(1) = -y(1) + 2*e + A - 2          (*)
#
# Введём новые начальные условия для метода стрельбы:
#
# y(0) = m                        
# y'(0) = y(0) - A = m - A             (**)
#
# m -- это параметр, который должен подобрать наш алгоритм.
# 
# Чтобы y(x, m) -- решение диффура с н.у. (**) было решением диффура с исходными н.у., 
# нужно так подобрать m, чтобы выполнялось то начальное условие, которое мы выбросили при переходе к (**):
# 
# Т.е. y'(1, m) == -y(1, m) + 2*e + A - 2
#
# Алгоритм устроен так:
# 
# Берём и методом дихотомии вычисляет такое M, при котором:
#
# y'(1, M) + y(1, M) - 2*e - A + 2 == 0
#
# Где y(x, m) -- решение диффура с н.у. (**)
#
# Тогда y(x, M) -- решение диффура с н.у. (*)


COLORS = [
    "#2196F3",
    "#4CAF50",
    "#F44336",
    "#00BCD4",
    "#9C27B0",
    "#FFC107",
    "#795548",
    "#673AB7",
]


class Method:
    color = 0

    def __init__(self, name, function):
        self.name = name
        self.function = function
        self.color = COLORS[Method.color % len(COLORS)]
        Method.color += 1

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


def method(name):
    return lambda function: Method(name, function)


def solve(method, f, x, y0, dy0):
    y = [y0]
    dy = [dy0]
    ddy = [f(x[0], y0, dy0)]
    for n in range(len(x) - 1):
        h = x[n + 1] - x[n]
        y_next, dy_next = method(f, n, h, x, y, dy)
        y.append(y_next)
        dy.append(dy_next)
        ddy.append(f(x[n + 1], y_next, dy_next))
    return y, dy, ddy


def shot(method, m):
    return solve(method, F, X, m, m - A)


def fire(method, m):
    y, dy, ddy = shot(method, m)
    return dy[N] + y[N] - 2*np.e - A + 2 


def dichotomy(a, b, f):
    fa = f(a)
    fb = f(b)
    
    if fa == 0:
        return a
    
    if fb == 0:
        return b
    
    if fa * fb > 0:
        raise ValueError()
    
    while abs(a - b) > EPSILON:
        c = (a + b) / 2
        fc = f(c)
        
        if fc == 0:
            return c
        
        if fc * fa < 0:
            b = c
            fb = fc
            continue
            
        if fc * fb < 0:
            a = c
            fa = fc
            continue
            
        raise ValueError()
    
    return (a + b) / 2


@method("Forward Euler's")
def explicit_euler(f, n, h, x, y, dy):
    dy_next = dy[n] + h * f(x[n], y[n], dy[n])
    y_next = y[n] + h * dy[n]
    return y_next, dy_next


@method("Semi-implicit Euler's")
def recount_euler(f, n, h, x, y, dy):
    xn = x[n]
    yn = y[n]
    dyn = dy[n]
    dy_next = dyn + h/2 * (f(xn, yn, dyn) + f(xn + h, yn + h * dyn, dyn + h * f(xn, yn, dyn)))
    y_next = yn + h/2 * (dyn + dyn + h * f(xn, yn, dyn))
    return y_next, dy_next


@method("Runge-Kutta's")
def runge_kutta(f, n, h, x, y, dy):
    xn = x[n]
    yn = y[n]
    dyn = dy[n]
    
    k1y = h * dyn
    k1dy = h * f(xn, yn, dyn)
    
    k2y = h * (dyn + k1dy/2)
    k2dy = h * f(xn + h/2, yn + k1y/2, dyn + k1dy/2)
    
    k3y = h * (dyn + k2dy/2)
    k3dy = h * f(xn + h/2, yn + k2y/2, dyn + k2dy/2)
    
    k4y = h * (dyn + k3dy)
    k4dy = h * f(xn + h, yn + k3y, dyn + k3dy)
    
    y_next = yn + (k1y + 2*(k2y + k3y) + k4y) / 6
    dy_next = dyn + (k1dy + 2*(k2dy + k3dy) + k4dy) / 6
    
    return y_next, dy_next


@method("Tridiagonal method 1")
def tridiag1(xs):
    N = len(xs) - 1
    h = (X[N] - X[0]) / N

    p = lambda x: 1
    q = lambda x: A*x*(1-x) + 2*A + 2

    alpha0 = 1
    alpha1 = -A

    beta0 = -1
    beta1 = 2*np.e + A - 2

    lambdas = np.zeros((N + 2,), dtype=np.float64)
    lambdas[1] = 1 / (1 + h*alpha0)

    mus = np.zeros((N + 2,), dtype=np.float64)
    mus[1] = -h * alpha1 / (1 + h*alpha0)

    for n in range(1, N+1):
        Xn = X[n]
        An = 2 + p(Xn) * h**2
        Bn = q(Xn) * h**2
        lambdas[n + 1] = -1 / (lambdas[n] - An)
        mus[n + 1] = (Bn - mus[n]) / (lambdas[n] - An)

    y = np.zeros(X.shape, dtype=np.float64)

    lambdaN = 1 - h * beta0
    muN = -h * beta1

    y[N] = -(mus[N] - muN) / (lambdas[N] - lambdaN)

    for n in range(N, 0, -1):
        y[n-1] = lambdas[n] * y[n] + mus[n]

    return y


@method("Tridiagonal method 2")
def tridiag2(xs):
    N = len(xs) - 1
    h = (X[N] - X[0]) / N

    p = lambda x: 1
    q = lambda x: A*x*(1-x) + 2*A + 2

    alpha0 = 1
    alpha1 = -A

    beta0 = -1
    beta1 = 2*np.e + A - 2

    k = 2*h*alpha0 + p(X[0])*h**2 + 2

    lambdas = np.zeros((N + 2,), dtype=np.float64)
    lambdas[1] = 2 / k

    mus = np.zeros((N + 2,), dtype=np.float64)
    mus[1] = -(2*h*alpha1 + q(X[0])*h**2) / k

    for n in range(1, N+1):
        Xn = X[n]
        An = 2 + p(Xn) * h**2
        Bn = q(Xn) * h**2
        lambdas[n + 1] = -1 / (lambdas[n] - An)
        mus[n + 1] = (Bn - mus[n]) / (lambdas[n] - An)

    y = np.zeros(X.shape, dtype=np.float64)

    lambdaN = h**2 * p(X[N]) / 2 - h*beta0 + 1
    muN = h**2 * q(X[N]) / 2 - h*beta1

    y[N] = -(mus[N] - muN) / (lambdas[N] - lambdaN)

    for n in range(N, 0, -1):
        y[n-1] = lambdas[n] * y[n] + mus[n]

    return y


METHODS = [
    explicit_euler,
    recount_euler,
    runge_kutta,
    tridiag1,
    tridiag2,
]


SELECTED_METHODS = set(METHODS)


def setup_ui():
    fig, ax = plt.subplots()
    plt.subplots_adjust(top=0.98, right=0.99, left=0.25, bottom=0.16)

    slider_axes = plt.axes([0.08, 0.04, 0.82, 0.03])
    slider = Slider(slider_axes, 'Steps', STEP_MIN, STEP_MAX, valinit=STEP)

    def slider_listener(value):
        global STEP
        STEP = int(value)
        plot(ax, value)
        fig.canvas.draw_idle()

    slider.on_changed(slider_listener)

    checkbox_axes = plt.axes([0.01, 0.3, 0.15, 0.5])
    checkbox = CheckButtons(checkbox_axes, METHODS, 
        [method in SELECTED_METHODS for method in METHODS])

    def checkbox_listener(label):
        method = [x for x in METHODS if x.name == label][0]
        if method in SELECTED_METHODS:
            SELECTED_METHODS.remove(method)
        else:
            SELECTED_METHODS.add(method)
        plot(ax, STEP)
        fig.canvas.draw_idle()

    checkbox.on_clicked(checkbox_listener)

    return ax, slider, checkbox


def plot(axes, steps_count):
    global X, N
    X = domain(steps_count)
    N = len(X) - 1

    axes.cla()
    axes.set_xlim([0, 1])
    axes.set_ylim([-10, 3])

    axes.plot(XX, Y(XX), "black")
    lines = ["Precise solution"]

    for method in METHODS:
        if method in SELECTED_METHODS:
            lines.append(method.name)

            if method == tridiag1 or method == tridiag2:
                axes.plot(X, method(X), method.color)
                continue

            M = dichotomy(-10, 10, lambda m: fire(method, m))

            y, dy, ddy = shot(method, M)

            axes.plot(X, y, method.color)

    axes.legend(lines, loc=1)


if __name__ == "__main__":
    ax, slider, checkbox = setup_ui()
    plot(ax, STEP)
    plt.show()
