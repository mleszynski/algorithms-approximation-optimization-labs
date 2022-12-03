# interior_point_quadratic.py
"""Volume 2: Interior Point for Quadratic Programs.
Marcelo Leszynski
Math 323 Section 3
26 March 2021
"""

import numpy as np
from scipy import linalg as la
from scipy.sparse import spdiags
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from cvxopt import matrix, solvers

def startingPoint(G, c, A, b, guess):
    """
    Obtain an appropriate initial point for solving the QP
    .5 x^T Gx + x^T c s.t. Ax >= b.
    Parameters:
        G -- symmetric positive semidefinite matrix shape (n,n)
        c -- array of length n
        A -- constraint matrix shape (m,n)
        b -- array of length m
        guess -- a tuple of arrays (x, y, mu) of lengths n, m, and m, resp.
    Returns:
        a tuple of arrays (x0, y0, l0) of lengths n, m, and m, resp.
    """
    m,n = A.shape
    x0, y0, l0 = guess

    # Initialize linear system
    N = np.zeros((n+m+m, n+m+m))
    N[:n,:n] = G
    N[:n, n+m:] = -A.T
    N[n:n+m, :n] = A
    N[n:n+m, n:n+m] = -np.eye(m)
    N[n+m:, n:n+m] = np.diag(l0)
    N[n+m:, n+m:] = np.diag(y0)
    rhs = np.empty(n+m+m)
    rhs[:n] = -(G.dot(x0) - A.T.dot(l0)+c)
    rhs[n:n+m] = -(A.dot(x0) - y0 - b)
    rhs[n+m:] = -(y0*l0)

    sol = la.solve(N, rhs)
    dx = sol[:n]
    dy = sol[n:n+m]
    dl = sol[n+m:]

    y0 = np.maximum(1, np.abs(y0 + dy))
    l0 = np.maximum(1, np.abs(l0+dl))

    return x0, y0, l0


# Problems 1-2
def qInteriorPoint(Q, c, A, b, guess, niter=20, tol=1e-16, verbose=False):
    """Solve the Quadratic program min .5 x^T Q x +  c^T x, Ax >= b
    using an Interior Point method.

    Parameters:
        Q ((n,n) ndarray): Positive semidefinite objective matrix.
        c ((n, ) ndarray): linear objective vector.
        A ((m,n) ndarray): Inequality constraint matrix.
        b ((m, ) ndarray): Inequality constraint vector.
        guess (3-tuple of arrays of lengths n, m, and m): Initial guesses for
            the solution x and lagrange multipliers y and eta, respectively.
        niter (int > 0): The maximum number of iterations to execute.
        tol (float > 0): The convergence tolerance.

    Returns:
        x ((n, ) ndarray): The optimal point.
        val (float): The minimum value of the objective function.
    """
    m,n = A.shape
    top = np.hstack((Q, np.zeros((n, m)), -A.T))
    mid = np.hstack((A, -np.eye(m), np.zeros((m, m))))

    def F(x, y, mu):
        """ Solves for F(x,y,mu) using the formula given in the spec.

        Parameters:
            x      (n, 1) ndarray:  Current point in the system.
            y      (m, 1) ndarray:  Vector of slack variables.
            mu     (m, 1) ndarray:  Vector of Lagrangian multipliers.

        Returns:
            F ((2m+n, 1) ndarray):    The 1-d vector representation of the KKT conditions.
        """
        return np.concatenate((Q@x - A.T@mu + c, A@x - y - b, np.diag(y)@np.diag(mu)@np.ones(m)))

    def search_direction(x, y, mu, v):
        """Calcuates delta x, delta y, and delta mu

        Parameters:
            x   (n, ) ndarray): Current point in the system.
            y   (m, ) ndarray): Vector of Slack variables.
            mu  (m, ) ndarray): Vector of Lagrangian multipliers.
            v            float: Duality Measure of the problem.

        Returns:
            gradients ((2m+n, ) ndarray):   grad of x, grad of y, grad of mu
        """
        # initialize variables #################################################
        M = np.diag(mu)
        Y = np.diag(y)
        e = np.ones(m)

        # create DF matrix #####################################################
        bot = np.hstack((np.zeros((m, n)), M, Y))
        DF = np.vstack((top, mid, bot))

        F_vec = F(x, y, mu)
        c_param = np.concatenate((np.zeros(m+n), e*v/10))

        # solve for gradients ##################################################
        lu, piv = la.lu_factor(DF)
        return la.lu_solve((lu, piv), c_param - F_vec)


    def step_size(x, mu, gradients):
        """Calcuates the size of the next search step.

        Parameters:
            x   ((n, ) ndarray):            Current point in the system.
            mu  ((m, ) ndarray):            Slack variables for the dual problem.
            gradients ((2m+n, ) ndarray):   Gradient vector as given by get_dir()

        Returns:
            alpha : float   Step size for y and mu.
            delta : float   Step size for x.
        """
        # initialize variables #################################################
        y_grad  = np.array([i if i < 0 else np.nan for i in gradients[n:m+n]])
        mu_grad = np.array([i if i < 0 else np.nan for i in gradients[m+n:]])

        # check for nonnegative conditions #####################################
        B_max = 1
        if np.nansum(mu_grad) != 0:
            # calculate proper B_max ###########################################
            B_temp = -mu / mu_grad
            B_max = B_temp[np.nanargmin(B_temp)]

        d_max = 1
        if np.nansum(y_grad) != 0:
            # calculate proper D_max ###########################################
            d_temp = -y / y_grad
            d_max = d_temp[np.nanargmin(d_temp)]

        return min(min(1, 0.95*B_max), min(1, 0.95*d_max))

    # get starting point #######################################################
    x, y, mu = startingPoint(Q, c, A, b, guess)

    # begin optimization process ###############################################
    for i in range(niter):
        v = (y@mu)/m
        if v < tol:
            break

        # compute step direction and size ######################################
        gradients = search_direction(x, y, mu, v)
        alpha = step_size(x, mu, gradients)

        # compute next iteration ###############################################
        x_grad = gradients[:n]
        y_grad = gradients[n:m+n]
        mu_grad = gradients[m+n:]

        x = x+alpha*x_grad
        y = y+alpha*y_grad
        mu = mu+alpha*mu_grad

    return x, .5*x@Q@x + c@x


def laplacian(n):
    """Construct the discrete Dirichlet energy matrix H for an n x n grid."""
    data = -1*np.ones((5, n**2))
    data[2,:] = 4
    data[1, n-1::n] = 0
    data[3, ::n] = 0
    diags = np.array([-n, -1, 0, 1, n])
    return spdiags(data, diags, n**2, n**2).toarray()

# Problem 3
def circus(n=15):
    """Solve the circus tent problem for grid size length 'n'.
    Display the resulting figure.
    """
    # create the tent pole configuration #######################################
    L = np.zeros((n,n))
    L[n//2-1:n//2+1,n//2-1:n//2+1] = .5
    m = [n//6-1, n//6, int(5*(n/6.))-1, int(5*(n/6.))]
    mask1, mask2 = np.meshgrid(m, m)
    L[mask1, mask2] = .3
    L = L.ravel()

    # set initial guesses ######################################################
    x = np.ones((n,n)).ravel()
    y = np.ones(n**2)
    mu = np.ones(n**2)

    # initialize variables #####################################################
    H = laplacian(n)
    A= np.eye(H.shape[0])
    c = np.ones(H.shape[0]) * 1 /((n-1)**2)

    # solve the problem ########################################################
    z = qInteriorPoint(H, c, A, L, (x,y,mu))[0].reshape((n,n))

    # plot the solution ########################################################
    domain = np.arange(n)
    X, Y = np.meshgrid(domain, domain)
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.plot_surface(X, Y, z, rstride=1, cstride=1, color='r')
    plt.show()


# Problem 4
def portfolio(filename="portfolio.txt"):
    """Markowitz Portfolio Optimization

    Parameters:
        filename (str): The name of the portfolio data file.

    Returns:
        (ndarray) The optimal portfolio with short selling.
        (ndarray) The optimal portfolio without short selling.
    """
    # read in data and initialize values #######################################
    data = np.loadtxt(filename)
    data = data[:,1:]
    n = data.shape[1]
    P = matrix(np.cov(data.T))
    q = matrix(np.zeros(n))
    G = matrix(-np.eye(n))
    h = matrix(np.zeros(n))
    A = matrix(np.vstack((np.ones(n), np.mean(data, axis=0))))
    b = matrix(np.array([1,1.13]))

    # find optimal portfolio with shortselling #################################
    solpos = solvers.qp(P,q,G,h,A,b)

    # find optimal portfolios without shortselling #############################
    G = matrix(np.zeros_like(P))
    solneg = solvers.qp(P,q,G,h,A,b)

    return np.array(solpos['x'])[:,0], np.array(solneg['x'])[:,0]


if __name__ == "__main__":
    # test problems 1 and 2 ####################################################
    Q = np.array([[1,-1],[-1,2]])
    c = np.array([-2,-6])
    A = np.array([[-1,-1],[1,-2],[-2,-1],[1,0],[0,1]])
    b = np.array([-2,-2,-3,0,0])
    x = np.array([.5,.5])
    y = np.ones(5)
    mu = np.ones(5)
    guess = (x, y, mu)
    print(qInteriorPoint(Q, c, A, b, guess))
    ############################################################################


    # test problem 3 ###########################################################
    circus()
    ############################################################################


    # test problem 4 ###########################################################
    print(portfolio())
    ############################################################################
