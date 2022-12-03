# gradient_methods.py
"""Volume 2: Gradient Descent Methods.
Marcelo Leszynski
Math 323 Sec 003
02/25/27
"""
import numpy as np
import math
from scipy import optimize as opt
from scipy import linalg as la
from matplotlib import pyplot as plt

# Problem 1
def steepest_descent(f, Df, x0, tol=1e-5, maxiter=100):
    """Compute the minimizer of f using the exact method of steepest descent.

    Parameters:
        f (function): The objective function. Accepts a NumPy array of shape
            (n,) and returns a float.
        Df (function): The first derivative of f. Accepts and returns a NumPy
            array of shape (n,).
        x0 ((n,) ndarray): The initial guess.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        ((n,) ndarray): The approximate minimum of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    # initialize variables ################################################
    x_k = x0

    # create iteration loop ###############################################
    for i in range(maxiter):
        # calculate the next step size ####################################
        step = lambda x : f(x_k - x * Df(x_k).T)
        alpha = opt.minimize_scalar(step).x
        

        # calculate the next x_k ##########################################
        x_k = x_k - alpha * Df(x_k).T

        # check for terminating condition #################################
        if la.norm(Df(x_k), np.inf) < tol:
            return x_k, True, i+1

    # create condition where maxiter is exceeded ##########################
    return x_k, False, maxiter

# Problem 2
def conjugate_gradient(Q, b, x0, tol=1e-4):
    """Solve the linear system Qx = b with the conjugate gradient algorithm.

    Parameters:
        Q ((n,n) ndarray): A positive-definite square matrix.
        b ((n, ) ndarray): The right-hand side of the linear system.
        x0 ((n,) ndarray): An initial guess for the solution to Qx = b.
        tol (float): The convergence tolerance.

    Returns:
        ((n,) ndarray): The solution to the linear system Qx = b.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    # initialize variables #####################################################
    r_k = (Q@x0) - b
    d_k = -r_k
    x_k = x0
    n = np.linalg.matrix_rank(Q)

    # construct while loop and calculate minimizer #############################
    for i in range(n):
        r_k0 = r_k
        d_k0 = d_k

        alpha = (r_k0@r_k0) / (d_k0@Q@d_k0)
        x_k = x_k + alpha * d_k0
        r_k = r_k0 + alpha * Q@d_k0
        beta = (r_k@r_k) / (r_k0@r_k0)
        d_k = -r_k + beta*d_k0

        # check for convergence condition ######################################
        if la.norm(r_k, np.inf) < tol:
            return x_k, True, i+1

    return x_k, False, n

# Problem 3
def nonlinear_conjugate_gradient(f, df, x0, tol=1e-5, maxiter=100):
    """Compute the minimizer of f using the nonlinear conjugate gradient
    algorithm.

    Parameters:
        f (function): The objective function. Accepts a NumPy array of shape
            (n,) and returns a float.
        Df (function): The first derivative of f. Accepts and returns a NumPy
            array of shape (n,).
        x0 ((n,) ndarray): The initial guess.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        ((n,) ndarray): The approximate minimum of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    # initialize variables #####################################################
    r_k = -df(x0).T
    d_k = r_k
    x_k = x0

    # construct while loop and calculate minimizer #############################
    for i in range(maxiter):
        r_k0 = r_k
        d_k0 = d_k

        r_k = -df(x_k).T
        beta = (r_k@r_k) / (r_k0@r_k0)
        d_k = r_k + beta * d_k0

        step = lambda x : f(x_k + x * d_k)
        res = opt.minimize_scalar(step)
        alpha = res.x

        x_k = x_k + alpha * d_k

        if la.norm(r_k) < tol:
            return x_k, True, i+1
    
    return x_k, False, maxiter



        # check for convergence condition ######################################


# Problem 4
def prob4(filename="linregression.txt",
          x0=np.array([-3482258, 15, 0, -2, -1, 0, 1829])):
    """Use conjugate_gradient() to solve the linear regression problem with
    the data from the given file, the given initial guess, and the default
    tolerance. Return the solution to the corresponding Normal Equations.
    """
    # initialize variables ##################################################
    A = np.loadtxt(filename)
    b = A[:,0].T
    A[:,0] = 1.

    # solve system ##########################################################
    return conjugate_gradient(A.T@A, A.T@b, x0)[0]

# Problem 5
class LogisticRegression1D:
    """Binary logistic regression classifier for one-dimensional data."""

    def fit(self, x, y, guess):
        """Choose the optimal beta values by minimizing the negative log
        likelihood function, given data and outcome labels.

        Parameters:
            x ((n,) ndarray): An array of n predictor variables.
            y ((n,) ndarray): An array of n outcome variables.
            guess (array): Initial guess for beta.
        """
        f = lambda b : np.sum([math.log(1+math.e**(-b[0]-b[1]*x[i]))+(1-y[i])*(b[0]+b[1]*x[i]) for i in range(len(x))])
        self.b_0, self.b_1 = opt.fmin_cg(f,guess)


    def predict(self, x):
        """Calculate the probability of an unlabeled predictor variable
        having an outcome of 1.

        Parameters:
            x (float): a predictor variable with an unknown label.
        """
        return 1/(1+math.e**(-(self.b_0+self.b_1*x)))


# Problem 6
def prob6(filename="challenger.npy", guess=np.array([20., -1.])):
    """Return the probability of O-ring damage at 31 degrees Farenheit.
    Additionally, plot the logistic curve through the challenger data
    on the interval [30, 100].

    Parameters:
        filename (str): The file to perform logistic regression on.
                        Defaults to "challenger.npy"
        guess (array): The initial guess for beta.
                        Defaults to [20., -1.]
    """
    # load in data and create object ###########################################
    A = np.load(filename)
    x = A[:,0].T
    y = A[:,1].T
    log_reg = LogisticRegression1D()
    log_reg.fit(x,y,guess)
    prob_31 = log_reg.predict(31)

    # plot info ################################################################
    domain = np.linspace(30,100,200)
    plt.plot(domain, log_reg.predict(domain), color='orange')
    plt.plot(x, y, 'bo', label='Previous Damage')
    plt.plot(31,prob_31,'go', label='P(Damage) at Launch')
    plt.xlabel('Temperature')
    plt.ylabel('O-Ring Damage')
    plt.legend(loc='upper right')
    plt.show()

    return prob_31


#if __name__ == "__main__":
    # test prob1() with easy function ##########################################
    #f = lambda x : x[0]**4 + x[1]**4 + x[2]**4
    #Df = lambda x : np.array([4*x[0]**3, 4*x[1]**3, 4*x[2]**3])
    #print(steepest_descent(f, Df, [1,1,1]))
    ############################################################################


    # test prob1() with rosenbrock #############################################
    #rosen = lambda x : (1-x[0])**2 + 100 * (x[1] - x[0]**2)**2
    #d_rosen = lambda x : np.array([-2*(1-x[0]) - 400 * x[0] * (x[1] - x[0]**2), 200 * (x[1] - x[0]**2)])
    #print(steepest_descent(rosen, d_rosen, [5,5], maxiter=10000))
    ############################################################################


    # test prob2() #############################################################
    #Q = np.array([[2,0], [0,4]])
    #b = np.array([1,8])
    #print(conjugate_gradient(Q,b,[5,5]))
    ############################################################################


    # test prob2() randomly ####################################################
    #n = 4
    #A = np.random.random((n,n))
    #Q = A.T @ A
    #b, x0 = np.random.random((2,n))

    #x = conjugate_gradient(Q,b,x0)
    #print(np.allclose(Q@x[0], b))
    ############################################################################


    # test prob3() #############################################################
    #print(opt.fmin_cg(opt.rosen, np.array([10,10]), fprime=opt.rosen_der))
    #rosen = lambda x : (1-x[0])**2 + 100 * (x[1] - x[0]**2)**2
    #d_rosen = lambda x : np.array([-2*(1-x[0]) - 400 * x[0] * (x[1] - x[0]**2), 200 * (x[1] - x[0]**2)])
    #print(nonlinear_conjugate_gradient(rosen, d_rosen, [10,10], maxiter = 1000))
    ############################################################################


    # test prob4() #############################################################
    #print(prob4())
    ############################################################################


    # test prob5() and prob6() using prob6() ###################################
    #prob6()
    ############################################################################