# matplotlib_intro.py
"""Python Essentials: Intro to Matplotlib.
Marcelo Leszynski
Math 321 Sec 005
12/9/20
"""

import numpy as np
from matplotlib import pyplot as plt


# Problem 1
def var_of_means(n):
    """Construct a random matrix A with values drawn from the standard normal
    distribution. Calculate the mean value of each row, then calculate the
    variance of these means. Return the variance.

    Parameters:
        n (int): The number of rows and columns in the matrix A.

    Returns:
        (float) The variance of the means of each row.
    """
    normal_array = np.random.normal(size = (n, n))
    mean = np.mean(normal_array, axis = 0)
    return float(np.var(mean))

def prob1():
    """Create an array of the results of var_of_means() with inputs
    n = 100, 200, ..., 1000. Plot and show the resulting array.
    """
    outputs = []
    iterator = [i * 100 for i in range(1,11)]
    for i in iterator:
        outputs.append(var_of_means(i))
    plt.plot(iterator, outputs)
    plt.show()


# Problem 2
def prob2():
    """Plot the functions sin(x), cos(x), and arctan(x) on the domain
    [-2pi, 2pi]. Make sure the domain is refined enough to produce a figure
    with good resolution.
    """
    x = np.linspace(-2 * np.pi, 2 * np.pi, 100)
    plt.plot(x, np.sin(x))
    plt.plot(x, np.cos(x))
    plt.plot(x, np.arctan(x))
    plt.show()


# Problem 3
def prob3():
    """Plot the curve f(x) = 1/(x-1) on the domain [-2,6].
        1. Split the domain so that the curve looks discontinuous.
        2. Plot both curves with a thick, dashed magenta line.
        3. Set the range of the x-axis to [-2,6] and the range of the
           y-axis to [-6,6].
    """
    first_half = np.linspace(-2, .999, 50)
    second_half = np.linspace(1.00001, 6, 50)
    plt.plot(first_half, 1/(first_half - 1), "m--", linewidth = 4)
    plt.plot(second_half, 1/(second_half - 1), "m--", linewidth = 4)
    plt.xlim(-2,6)
    plt.ylim(-6,6)
    plt.show()

# Problem 4
def prob4():
    """Plot the functions sin(x), sin(2x), 2sin(x), and 2sin(2x) on the
    domain [0, 2pi].
        1. Arrange the plots in a square grid of four subplots.
        2. Set the limits of each subplot to [0, 2pi]x[-2, 2].
        3. Give each subplot an appropriate title.
        4. Give the overall figure a title.
        5. Use the following line colors and styles.
              sin(x): green solid line.
             sin(2x): red dashed line.
             2sin(x): blue dashed line.
            2sin(2x): magenta dotted line.
    """
    domain = np.linspace(0, 2 * np.pi, 200)
    plt.subplot(221)
    plt.plot(domain, np.sin(domain), "g-")
    plt.title("sin(x)", fontsize = 12)
    plt.subplot(222)
    plt.plot(domain, np.sin(domain * 2), "r--")
    plt.title("sin(2x)", fontsize = 12)
    plt.subplot(223)
    plt.plot(domain, 2 * np.sin(domain), "b--")
    plt.title("2sin(x)", fontsize = 12)
    plt.subplot(224)
    plt.plot(domain, 2 * np.sin(domain * 2), "m:")
    plt.title("2sin(2x)", fontsize = 12)
    plt.axis([0, 2 * np.pi, -2, 2])
    plt.suptitle("Sine Function Transformations", fontsize = 18)
    plt.tight_layout()
    plt.show()

# Problem 5
def prob5():
    """Visualize the data in FARS.npy. Use np.load() to load the data, then
    create a single figure with two subplots:
        1. A scatter plot of longitudes against latitudes. Because of the
            large number of data points, use black pixel markers (use "k,"
            as the third argument to plt.plot()). Label both axes.
        2. A histogram of the hours of the day, with one bin per hour.
            Label and set the limits of the x-axis.
    """
    data = np.load("FARS.npy")
    plt.subplot(121)
    plt.plot(data[:,1], data[:,2], "k,")
    plt.axis("equal")
    plt.xlabel("Longitude", fontsize = 12)
    plt.ylabel("Latitude", fontsize = 12)
    plt.subplot(122)
    plt.hist(data[:,0], bins = np.arange(0, 25))
    plt.xlim(0,24)
    plt.xlabel("Hour of the Day", fontsize = 12)
    plt.show()

# Problem 6
def prob6():
    """Plot the function f(x,y) = sin(x)sin(y)/xy on the domain
    [-2pi, 2pi]x[-2pi, 2pi].
        1. Create 2 subplots: one with a heat map of f, and one with a contour
            map of f. Choose an appropriate number of level curves, or specify
            the curves yourself.
        2. Set the limits of each subplot to [-2pi, 2pi]x[-2pi, 2pi].
        3. Choose a non-default color scheme.
        4. Add a colorbar to each subplot.
    """
    x = np.linspace(-2 * np.pi, 2 * np.pi, 200)
    y = np.copy(x)
    X, Y = np.meshgrid(x, y)
    Z = (np.sin(X) * np.sin(Y)) / (X * Y)
    plt.subplot(121)
    plt.pcolormesh(X, Y, Z, cmap = "magma")
    plt.colorbar()
    plt.xlim(-2 * np.pi, 2 * np.pi)
    plt.ylim(-2 * np.pi, 2 * np.pi)
    plt.subplot(122)
    plt.contour(X, Y, Z, 50, cmap = "coolwarm")
    plt.colorbar()
    plt.xlim(-2 * np.pi, 2 * np.pi)
    plt.ylim(-2 * np.pi, 2 * np.pi)
    plt.show()