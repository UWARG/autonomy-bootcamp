import numpy as np
from matplotlib import pyplot as plt

def plot_graph(x, y, fmt: str = None, label: str = None, title: str = None, xLabel: str = None, yLabel: str = None, legend: bool = False):
    """
    Returns an image of a plot graph through pyplot using given parameters

    Parameters
    ----------
        x : array or scalar
            variable containing array to be included on x axis of plot
        y : array or scalar
            variable containing array to be included on y axis of plot
        fmt : str, optional
            variable containing formatting for plot lines such as type of line (dotted, dashed, solid, etc.) and color of lines 
        label : str, optional
            label for legend
        title : str, optional
            title of plotted graph
        xLabel : str, optional
            label of x-axis
        yLabel : str, optional
            label of y-axis
        legend : bool, optional
        
    Returns
    -------
        image of plotted graph
    """
    plt.plot(x, y, f"{fmt}", label=f"{label}")
    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    if legend:
        plt.legend()
    plt.show()