import numpy as np
from matplotlib import pyplot as plt

def plot_graph(x, y, fmt: str = None, label: str = None, title: str = None, xlabel: str = None, ylabel: str = None, legend: bool = False):
    plt.plot(x, y, f"{fmt}", label=f"{label}")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if (legend):
        plt.legend()
    # plt.figure()
    plt.show()