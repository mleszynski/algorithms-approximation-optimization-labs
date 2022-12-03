# interior_point_linear.py
"""Volume 2: Interior Point for Linear Programs.
Marcelo Leszynski
Math 323 Sec 003
04/13/21
"""

import numpy as np
from scipy import linalg as la
from scipy.stats import linregress
from matplotlib import pyplot as plt


# Auxiliary Functions ---------------------------------------------------------
def starting_point(A, b, c):
    """Calculate an initial guess to the solution of the linear program
    min c^T x, Ax = b, x>=0.
    Reference: Nocedal and Wright, p. 410.
    """
    # Calculate x, lam, mu of minimal norm satisfying both
    # the primal and dual constraints.
    B = la.inv(A @ A.T)
    x = A.T @ B @ b
    lam = B @ A @ c
    mu = c - (A.T @ lam)

    # Perturb x and s so they are nonnegative.
    dx = max((-3./2)*x.min(), 0)
    dmu = max((-3./2)*mu.min(), 0)
    x += dx*np.ones_like(x)
    mu += dmu*np.ones_like(mu)

    # Perturb x and mu so they are not too small and not too dissimilar.
    dx = .5*(x*mu).sum()/mu.sum()
    dmu = .5*(x*mu).sum()/x.sum()
    x += dx*np.ones_like(x)
    mu += dmu*np.ones_like(mu)

    return x, lam, mu

# Use this linear program generator to test your interior point method.
def randomLP(j,k):
    """Generate a linear program min c^T x s.t. Ax = b, x>=0.
    First generate m feasible constraints, then add
    slack variables to convert it into the above form.
    Parameters:
        j (int >= k): number of desired constraints.
        k (int): dimension of space in which to optimize.
    Returns:
        A ((j, j+k) ndarray): Constraint matrix.
        b ((j,) ndarray): Constraint vector.
        c ((j+k,), ndarray): Objective function with j trailing 0s.
        x ((k,) ndarray): The first 'k' terms of the solution to the LP.
    """
    A = np.random.random((j,k))*20 - 10
    A[A[:,-1]<0] *= -1
    x = np.random.random(k)*10
    b = np.zeros(j)
    b[:k] = A[:k,:] @ x
    b[k:] = A[k:,:] @ x + np.random.random(j-k)*10
    c = np.zeros(j+k)
    c[:k] = A[:k,:].sum(axis=0)/k
    A = np.hstack((A, np.eye(j)))
    return A, b, -c, x


# Problems --------------------------------------------------------------------
def interiorPoint(A, b, c, niter=20, tol=1e-16, verbose=False):
    """Solve the linear program min c^T x, Ax = b, x>=0
    using an Interior Point method.

    Parameters:
        A ((m,n) ndarray): Equality constraint matrix with full row rank.
        b ((m, ) ndarray): Equality constraint vector.
        c ((n, ) ndarray): Linear objective function coefficients.
        niter (int > 0): The maximum number of iterations to execute.
        tol (float > 0): The convergence tolerance.

    Returns:
        x ((n, ) ndarray): The optimal point.
        val (float): The minimum value of the objective function.
    """
    # implement F function #####################################################
    def get_f(temp_x, temp_lam, temp_mu):
        """Creates a vector representing the KKT conditions of a problem.

        Parameters:
            temp_x ((n, ) ndarray):     Current point in the system.
            temp_lam ((m, ) ndarray):   Variables of the dual problem.
            temp_mu ((n, ) ndarray):    Slack variables for the dual problem.

        Returns:
            F(x, lambda, mu) ((2n+m, ) ndarray):    The 1-d vector representation 
                                                    of the KKT conditions.
        """
        top = (A.T@temp_lam) + temp_mu - c
        mid = (A@temp_x) - b
        bottom = np.diag(temp_mu)@temp_x
        
        # return the array mentioned on pg 3 ###################################
        return np.concatenate((top, mid, bottom))

    # compute the first two block rows of df ###################################
    tz_dimension = A.shape[1]
    t_zeros = np.zeros((tz_dimension, tz_dimension))
    I = np.eye(tz_dimension)
    t_row = np.hstack((t_zeros, A.T, I))  # top block row of DF

    m_zeros = np.zeros((A.shape[0], t_row.shape[1] - A.shape[1]))
    m_row = np.hstack((A, m_zeros))  # mid block row of DF
    df = np.vstack((t_row, m_row))  # top two block rows of DF

    # compute the search direction #############################################
    def get_grad(temp_x, temp_lam, temp_mu, v):
        """Calcuates the search direction.

        Parameters:
            temp_x ((n, ) ndarray):     Current point in the system.
            temp_lam ((m, ) ndarray):   Variables of the dual problem.
            temp_mu ((n, ) ndarray):    Slack variables for the dual problem.
            v : float                   Duality Measure of the problem.

        Returns:
            gradients ((2n+m, ) ndarray):   A 1-d vector where the first n terms
                                            are the gradient of x, the next m terms 
                                            are the gradient of lambda, and the 
                                            final n terms are the gradient of mu.
        """
        # create DF Matrix bottom row ##########################################
        M = np.diag(temp_mu)
        X = np.diag(temp_x)
        b_zeros = np.zeros((M.shape[0], df.shape[1]-(M.shape[1]+X.shape[1])))
        b_row = np.hstack((M, b_zeros, X))
        df_complete = np.vstack((df, b_row))

        # compute -F vector and add centering parameter ########################
        neg_f = -get_f(temp_x, temp_lam, temp_mu)
        c_param = np.zeros(len(temp_x)+len(temp_lam))
        c_param = np.concatenate((c_param, np.array([v/10]*len(temp_mu))))

        # solve and return system ##############################################
        lu, piv = la.lu_factor(df_complete)
        x = la.lu_solve((lu, piv), neg_f+c_param)

        return x

    # calculate the stepsizes ##################################################
    def get_step_size(temp_x, temp_mu, gradients):
        """Calcuates the size of the next search step.

        Parameters:
            temp_x ((n, ) ndarray):     Current point in the system.
            temp_mu ((n, ) ndarray):    Slack variables for the dual problem.
            gradients ((2n+m, ) ndarray):   Gradient vector as given by get_dir()

        Returns:
            alpha : float   Step size for lambda and mu.
            delta : float   Step size for x.
        """
        # strip out individual gradient from gradients #########################
        x_grad = np.array(gradients[:len(temp_x)])
        mu_grad = np.array(gradients[len(temp_x)+(len(gradients)-2*len(temp_x)):])

        # mask gradients #######################################################
        x_grad = np.array([grad if grad < 0 else np.nan for grad in x_grad])
        mu_grad = np.array([grad if grad < 0 else np.nan for grad in mu_grad])

        # check for nonnegative conditions #####################################
        a_max = 1
        if np.nansum(mu_grad) != 0:
            # calculate proper a_max ###########################################
            a_temp = -temp_mu/mu_grad
            a_max = a_temp[np.nanargmin(a_temp)]
        
        d_max = 1
        if np.nansum(x_grad) != 0:
            # calculate proper d_max ###########################################
            d_temp = -temp_x/x_grad
            d_max = d_temp[np.nanargmin(d_temp)]

        return min(1, 0.95*a_max), min(1, 0.95*d_max)


    # begin optimization process ###############################################
    x, lam, mu = starting_point(A, b, c)
    for i in range(niter):
        # calculate the duality measure and check for completion ###############
        v = (x@mu)/len(x)
        if v < tol:
            break

        # compute step direction and size ######################################
        gradients = get_grad(x, lam, mu, v)
        alpha, delta = get_step_size(x, mu, gradients)

        # compute next iteration ###############################################
        x_grad = gradients[:len(x)]
        lam_grad = gradients[len(x):len(x)+len(lam)]
        mu_grad = gradients[len(x)+len(lam):]

        x = x+delta*x_grad
        lam = lam+alpha*lam_grad
        mu = mu+alpha*mu_grad

    return x, c@x


def leastAbsoluteDeviations(filename='simdata.txt'):
    """Generate and show the plot requested in the lab."""
    # read in and initialize data ##############################################
    data = np.loadtxt(filename)
    m = data.shape[0]
    n = data.shape[1] - 1
    c = np.zeros(3*m + 2*(n + 1))
    c[:m] = 1
    y = np.empty(2*m)
    y[::2] = -data[:, 0]
    y[1::2] = data[:, 0]
    x = data[:, 1:]

    # initialize the constraint matrix #########################################
    A = np.ones((2*m, 3*m + 2*(n + 1)))
    A[::2, :m] = np.eye(m)
    A[1::2, :m] = np.eye(m)
    A[::2, m:m+n] = -x
    A[1::2, m:m+n] = x
    A[::2, m+n:m+2*n] = x
    A[1::2, m+n:m+2*n] = -x
    A[::2, m+2*n] = -1
    A[1::2, m+2*n+1] = -1
    A[:, m+2*n+2:] = -np.eye(2*m, 2*m)

    # calculate and extract the solution
    sol = interiorPoint(A, y, c)[0]
    beta = sol[m:m+n] - sol[m+n:m+2*n]
    b = sol[m+2*n] - sol[m+2*n+1]

    # plot least squares solution ##############################################
    slope, intercept = linregress(data[:,1], data[:,0])[:2]
    domain = np.linspace(0,10,200)
    plt.subplot(211)
    plt.plot(domain, domain*beta + b)
    plt.plot(data[:,1], data[:,0], 'ok')
    plt.title('Least Absolute Deviation')

    # plot LAD solution ########################################################
    plt.subplot(212)
    plt.plot(domain, domain*slope + intercept, 'g-')
    plt.plot(data[:,1], data[:,0], 'ok')
    plt.title('Least Squares')
    plt.tight_layout()
    plt.show()


#if __name__ == "__main__":
    # test problems 1-4 ########################################################
    #j, k = 7,5
    #A, b, c, x = randomLP(j, k)
    #point, value = interiorPoint(A, b, c)
    #print(np.allclose(x, point[:k]))
    ############################################################################


    # test problem 5 ###########################################################
    #leastAbsoluteDeviations('simdata.txt')