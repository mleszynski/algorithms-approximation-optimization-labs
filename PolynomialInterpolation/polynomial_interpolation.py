# polynomial_interpolation.py
"""Volume 2: Polynomial Interpolation.
Marcelo Leszynski
Math 323 Section 3
02/25/21
"""

import numpy as np
from scipy.interpolate import BarycentricInterpolator
from matplotlib import pyplot as plt

# Problems 1 and 2
def lagrange(xint, yint, points):
    """Find an interpolating polynomial of lowest degree through the points
    (xint, yint) using the Lagrange method and evaluate that polynomial at
    the specified points.

    Parameters:
        xint ((n,) ndarray): x values to be interpolated.
        yint ((n,) ndarray): y values to be interpolated.
        points((m,) ndarray): x values at which to evaluate the polynomial.

    Returns:
        ((m,) ndarray): The value of the polynomial at the specified points.
    """

    matrix = []

    for j in range(len(xint)):
        # calculate the denominator ############################################
        denom = np.product([(xint[j]-xint[k]) for k in range(len(xint)) if j!=k])

        eval = []
        # calculate the numerator ##############################################
        for point in points:
            numer = np.product([(point-xint[k]) for k in range(len(xint)) if j!=k])
            eval.append(numer)

        eval = eval / denom
        matrix.append(yint[j]*eval)

    # convert the matrix to the right data type ################################
    matrix = np.array(matrix)

    # return sum of rows in matrix #############################################
    return matrix.sum(axis=0)

# Problems 3 and 4
class Barycentric:
    """Class for performing Barycentric Lagrange interpolation.

    Attributes:
        w ((n,) ndarray): Array of Barycentric weights.
        n (int): Number of interpolation points.
        x ((n,) ndarray): x values of interpolating points.
        y ((n,) ndarray): y values of interpolating points.
    """

    def __init__(self, xint, yint):
        """Calculate the Barycentric weights using initial interpolating points.

        Parameters:
            xint ((n,) ndarray): x values of interpolating points.
            yint ((n,) ndarray): y values of interpolating points.
        """
        self._x = xint
        self._y = yint

        # create an array of Barycentric weights ###############################
        n = len(xint)
        self._w = np.ones(n)

        C = (np.max(xint) - np.min(xint)) / 4
        shuffle = np.random.permutation(n-1)

        for i in range(n):
            temp = (xint[i] - np.delete(xint, i)) / C
            temp = temp[shuffle]
            self._w[i] /= np.product(temp)

    def __call__(self, points):
        """Using the calcuated Barycentric weights, evaluate the interpolating polynomial
        at points.

        Parameters:
            points ((m,) ndarray): Array of points at which to evaluate the polynomial.

        Returns:
            ((m,) ndarray): Array of values where the polynomial has been computed.
        """
        point_vals = []
        for point in points:
            if point in self._x:
                point_vals.append(self._y[np.where(point == self._x)[0]])
            else:
                top = sum([self._w[i] *self._y[i]/(point-self._x[i]) for i in range(len(self._y))])
                bottom = sum([self._w[i]/(point-self._x[i]) for i in range(len(self._y))])
                point_vals.append(top/bottom)
        return np.array(point_vals, dtype=object)

    # Problem 4
    def add_weights(self, xint, yint):
        """Update the existing Barycentric weights using newly given interpolating points
        and create new weights equal to the number of new points.

        Parameters:
            xint ((m,) ndarray): x values of new interpolating points.
            yint ((m,) ndarray): y values of new interpolating points.
        """
        n = len(self._x)
        self._x = np.append(self._x,xint)
        self._y = np.append(self._y,yint)
        w_1 = np.ones(len(xint))

        for j,x_val in enumerate(xint):
            for i in range(n):
                self._w[i] /= (self._x[i] - x_val)

            for k in range(len(self._x)):
                if k != j:
                    w_1[j] *= 1/(self._x[j] - self._x[k])

        self._w = np.append(self._w,w_1)

# Problem 5
def prob5():
    """For n = 2^2, 2^3, ..., 2^8, calculate the error of intepolating Runge's
    function on [-1,1] with n points using SciPy's BarycentricInterpolator
    class, once with equally spaced points and once with the Chebyshev
    extremal points. Plot the absolute error of the interpolation with each
    method on a log-log plot.
    """
    # create equally space and chebyshev points ################################
    f = lambda x: 1/(1+25 * x**2)
    domain = np.linspace(-1, 1, 400)

    n_pow = [2**i for i in range(2,9)]
    b_error = []
    c_error = []

    for n in n_pow:
        pts = np.linspace(-1, 1, n)
        extrem = np.array([np.cos(j*np.pi/n) for j in range(n+1)])

        poly_b = BarycentricInterpolator(pts, f(pts))
        poly_c = BarycentricInterpolator(extrem, f(extrem))

        b_error.append(np.linalg.norm(f(domain)-poly_b(domain),ord=np.inf))
        c_error.append(np.linalg.norm(f(domain)-poly_c(domain),ord=np.inf))

    # plot results #############################################################
    plt.loglog(n_pow, b_error, label="Equally Spaced Points")
    plt.loglog(n_pow, c_error, label="Chebyshev Extremal Points")
    plt.legend(loc='upper left')
    plt.show()

# Problem 6
def chebyshev_coeffs(f, n):
    """Obtain the Chebyshev coefficients of a polynomial that interpolates
    the function f at n points.

    Parameters:
        f (function): Function to be interpolated.
        n (int): Number of points at which to interpolate.

    Returns:
        coeffs ((n+1,) ndarray): Chebyshev coefficients for the interpolating polynomial.
    """
    # use chebyshev coefficients ###############################################
    y = np.cos((np.pi * np.arange(2*n)) / n)
    samples = f(y)

    coeffs = np.real(np.fft.fft(samples))[:n+1] / n
    coeffs[0] = coeffs[0]/2
    coeffs[n] = coeffs[n]/2

    return coeffs

# Problem 7
def prob7(n):
    """Interpolate the air quality data found in airdata.npy using
    Barycentric Lagrange interpolation. Plot the original data and the
    interpolating polynomial.

    Parameters:
        n (int): Number of interpolating points to use.
    """
    # load file and initialize data ############################################
    data = np.load("airdata.npy")

    # implement spec algorithm #################################################
    f = lambda a, b, n: .5*(a+b + (b-a) * np.cos(np.arange(n+1) * np.pi / n))
    a = 0
    b = 366 - 1/24
    domain = np.linspace(0, b, 8784)
    points = f(a, b, n)
    temp = np.abs(points - domain.reshape(8784, 1))
    temp2 = np.argmin(temp, axis=0)
    poly = Barycentric(domain[temp2], data[temp2])

    # plot values ##############################################################
    plt.subplot(211)
    plt.plot(domain, data)
    plt.title("Original Data")
    plt.xlabel("Time")
    plt.ylabel("Air Quality")

    plt.subplot(212)
    plt.plot(domain, poly(domain))
    plt.title("Interpolation")
    plt.xlabel("Time")
    plt.ylabel("Air Quality")

    plt.tight_layout()
    plt.show()
