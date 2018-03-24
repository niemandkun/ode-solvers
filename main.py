#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons


EPSILON = 0.1E-10

STEP = 25

STEP_MIN = 5

STEP_MAX = 100

DOMAIN_START = 0

DOMAIN_END = 1

F = lambda x, y: -20 * y**2 * (x - 0.4)

Y = lambda x: 1 / (10 * x**2 - 8 * x + 2)

Y0 = 0.5

COLORS = [
    "#607D8B",
    "#2196F3",
    "#4CAF50",
    "#F44336",
    "#00BCD4",
    "#9C27B0",
    "#FFC107",
    "#795548",
    "#009688",
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


def domain(nodes_count):
    return np.linspace(DOMAIN_START, DOMAIN_END, nodes_count + 1)


def simple_iter(x, f):
    err = 100
    i = 0
    while err > EPSILON:
        xnew = f(x)
        err = abs(x - xnew)
        x = xnew
        i += 1
        if i > 1000:
            print("Warning: simple iteration method has failed. " + 
                    "Inaccurate result will be returned.")
            return x
    return x


def solve(method, f, x, y0):
    y = [y0]
    for n in range(len(x) - 1):
        h = x[n + 1] - x[n]
        y.append(method(f, n, h, x, y))
    return y


@method("Forward Euler's")
def explicit_euler(f, n, h, x, y):
    return y[n] + h * f(x[n], y[n])


@method("Semi-implicit Euler's")
def recount_euler(f, n, h, x, y):
    xn = x[n]
    yn = y[n]
    return yn + h/2 * (f(xn, yn) + f(xn + h, yn + h * f(xn, yn)))


@method("Cauchy's")
def cauchy(f, n, h, x, y):
    xn = x[n]
    yn = y[n]
    return yn + h * f(xn + h/2, yn + h/2 * f(xn, yn))


@method("Runge-Kutta's")
def runge_kutta(f, n, h, x, y):
    xn = x[n]
    yn = y[n]
    k1 = h * f(xn, yn)
    k2 = h * f(xn + h/2, yn + k1/2)
    k3 = h * f(xn + h/2, yn + k2/2)
    k4 = h * f(xn + h, yn + k3)
    return yn + (k1 + 2*(k2 + k3) + k4) / 6


@method("Backward Euler's")
def implicit_euler(f, n, h, x, y):
    yn = y[n]
    xn1 = x[n + 1]
    g = lambda y: yn + h * f(xn1, y)
    yn1 = simple_iter(yn, g)
    return g(yn1) 


@method("Adams's")
def adams(f, n, h, x, y):
    yn = y[n]
    xn = x[n]
    fn = f(xn, yn)
    xn1 = x[n + 1]
    g = lambda y: yn + h/2 * (fn + f(xn1, y))
    yn1 = simple_iter(yn, g)
    return g(yn1)


@method("Second order Tailor's")
def tailor2(f, n, h, x, y):
    xn = x[n]
    yn = y[n]
    fn = f(xn, yn)
    f_x = -20 * yn**2
    f_y = -40 * yn * (xn - 0.4)
    k1 = yn
    k2 = h * fn
    k3 = h**2 * (f_x + f_y * fn) / 2
    return k1 + k2 + k3


@method("Third order Tailor's")
def tailor3(f, n, h, x, y):
    xn = x[n]
    yn = y[n]
    fn = f(xn, yn)
    f_x = -20 * yn**2
    f_y = -40 * yn * (xn - 0.4)
    f_xx = 0
    f_xy = -40 * yn
    f_yy = -40 * (xn - 0.4)
    k1 = yn
    k2 = h * fn
    k3 = h**2 * (f_x + f_y * fn) / 2
    k4 = h**3 * (f_x + f_xx + f_xy + (f_y + f_yy + f_xy) * fn) / 6
    return k1 + k2 + k3 + k4


METHODS = [
    explicit_euler,
    implicit_euler,
    recount_euler,
    cauchy,
    runge_kutta,
    adams,
    tailor2,
    tailor3,
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
    xs = domain(steps_count)
    xs1 = domain(1000)

    axes.cla()
    axes.set_xlim([0, 1])
    axes.set_ylim([0, 3])

    axes.plot(xs1, Y(xs1), "black")
    lines = ["1 / y = 10*x^2 - 8*x + 2"]

    for method in METHODS:
        if method in SELECTED_METHODS:
            axes.plot(xs, solve(method, F, xs, Y0), method.color)
            lines.append(method.name)

    axes.legend(lines, loc=0)


if __name__ == "__main__":
    ax, slider, checkbox = setup_ui()
    plot(ax, STEP)
    plt.show()
