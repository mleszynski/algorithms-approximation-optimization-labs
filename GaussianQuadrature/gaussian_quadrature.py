# quassian_quadrature.py
"""Volume 2: Gaussian Quadrature.
Marcelo Leszynski
Math 323 Section 3
03/05/21
"""

import numpy as np
from scipy import linalg as la
from scipy.stats import norm
from scipy.integrate import nquad
from scipy.integrate import quad
from matplotlib import pyplot as plt


class GaussianQuadrature:
    """Class for integrating functions on arbitrary intervals using Gaussian
    quadrature with the Legendre polynomials or the Chebyshev polynomials.
    """
    # Problems 1 and 3
    def __init__(self, n, polytype="legendre"):
        """Calculate and store the n points and weights corresponding to the
        specified class of orthogonal polynomial (Problem 3). Also store the
        inverse weight function w(x)^{-1} = 1 / w(x).

        Parameters:
            n (int): Number of points and weights to use in the quadrature.
            polytype (string): The class of orthogonal polynomials to use in
                the quadrature. Must be either 'legendre' or 'chebyshev'.

        Raises:
            ValueError: if polytype is not 'legendre' or 'chebyshev'.
        """
        # check for correct input ##############################################
        if polytype == "legendre":
            self._w_inv = lambda x: 1
        elif polytype == "chebyshev":
            self._w_inv = lambda x: np.sqrt(1 - x**2)
        else:
            raise ValueError("Polytype needs to be either \'legendre\' or \'chebyshev\'")

        # store attributes #####################################################
        self._n = n
        self._polytype = polytype
        self._points, self._weights = self.points_weights(n)


    # Problem 2
    def points_weights(self, n):
        """Calculate the n points and weights for Gaussian quadrature.

        Parameters:
            n (int): The number of desired points and weights.

        Returns:
            points ((n,) ndarray): The sampling points for the quadrature.
            weights ((n,) ndarray): The weights corresponding to the points.
        """
        # calculate the nxn jacobian ###########################################
        points = []
        if self._polytype == "legendre":
            leg_beta_k = np.array([np.sqrt(k**2 / (4*k**2 - 1)) for k in range(1,n)])
            J_1 = np.diag(leg_beta_k,1)
            J_2 = np.diag(leg_beta_k,-1)
        else:
            cheb_beta_k = np.array([np.sqrt(1/2) if k==1 else (1/2) for k in range(1,n)])
            J_1 = np.diag(cheb_beta_k,1)
            J_2 = np.diag(cheb_beta_k,-1)
        jacobian = J_1 + J_2

        # append points ########################################################
        eigen_vals, eigen_vecs = la.eig(jacobian)
        eigen_vals = eigen_vals.real
        for val in eigen_vals:
            points.append(val)

        # calculate weights ####################################################
        if self._polytype == "legendre":
            weights = np.array([ eigen_vecs[0][i]**2 * 2 for i in range(n) ])
        else:
            weights = np.array([ eigen_vecs[0][i]**2 * np.pi for i in range(n) ])

        return np.array(points), weights


    # Problem 3
    def basic(self, f):
        """Approximate the integral of a f on the interval [-1,1]."""
        # define g function ####################################################
        g = lambda x: f(x)*self._w_inv(x)

        # dot g function with weight values ####################################
        return np.dot(g(self._points),self._weights)

    # Problem 4
    def integrate(self, f, a, b):
        """Approximate the integral of a function on the interval [a,b].

        Parameters:
            f (function): Callable function to integrate.
            a (float): Lower bound of integration.
            b (float): Upper bound of integration.

        Returns:
            (float): Approximate value of the integral.
        """
        # define h function ####################################################
        h = lambda x: f(((b-a)*x + (a+b))/2)

        # return h basic #######################################################
        return ((b-a)/2) * self.basic(h)

    # Problem 6.
    def integrate2d(self, f, a1, b1, a2, b2):
        """Approximate the integral of the two-dimensional function f on
        the interval [a1,b1]x[a2,b2].

        Parameters:
            f (function): A function to integrate that takes two parameters.
            a1 (float): Lower bound of integration in the x-dimension.
            b1 (float): Upper bound of integration in the x-dimension.
            a2 (float): Lower bound of integration in the y-dimension.
            b2 (float): Upper bound of integration in the y-dimension.

        Returns:
            (float): Approximate value of the integral.
        """
        # define h and g functions #############################################
        h = lambda x,y: f(((b1-a1)*x +(a1+b1))/2, ((b2-a2)*y +(a2+b2))/2)
        g = lambda x,y: h(x,y) * self._w_inv(x) * self._w_inv(y)

        # calculate and return the sum values ##################################
        sum_j = [np.dot(self._weights, g(xi,self._points)) * wi for xi,wi in zip(self._points,self._weights)]
        sum_i = sum(sum_j)

        return np.real(((b1-a1)*(b2-a2) / 4) * sum_i)

# Problem 5
def prob5():
    """Use scipy.stats to calculate the "exact" value F of the integral of
    f(x) = (1/sqrt(2 pi))e^((-x^2)/2) from -3 to 2. Then repeat the following
    experiment for n = 5, 10, 15, ..., 50.
        1. Use the GaussianQuadrature class with the Legendre polynomials to
           approximate F using n points and weights. Calculate and record the
           error of the approximation.
        2. Use the GaussianQuadrature class with the Chebyshev polynomials to
           approximate F using n points and weights. Calculate and record the
           error of the approximation.
    Plot the errors against the number of points and weights n, using a log
    scale for the y-axis. Finally, plot a horizontal line showing the error of
    scipy.integrate.quad() (which doesn’t depend on n).
    """
    # define f function ########################################################
    f = lambda x: np.e**((-x**2)/2) / np.sqrt(2*np.pi)

    # calculate the exact value ################################################
    exact_val = norm.cdf(2) - norm.cdf(-3)
    exact_error = abs(quad(f, -3, 2)[0] - exact_val)

    # calculate the error values ###############################################
    n_vals = range(5,51,5)
    leg_error = []
    cheb_error = []
    for n in n_vals:
        GQ = GaussianQuadrature(n)
        leg_error.append(abs(GQ.integrate(f, -3, 2) - exact_val))

        GQ = GaussianQuadrature(n, "chebyshev")
        cheb_error.append(abs(GQ.integrate(f, -3, 2) - exact_val))

    # plot results #############################################################
    plt.plot(n_vals, leg_error, label="Legendre")
    plt.plot(n_vals, cheb_error, label="Chebyshev")
    plt.plot([5,50], [exact_error, exact_error], label="Scipy Quads")
    plt.xlabel("Number of Points")
    plt.yscale("log")
    plt.ylabel("Error")
    plt.legend()
    plt.show()
