# cvxpy_intro.py
"""Volume 2: Intro to CVXPY.
Marcelo Leszynski
Math 323 Sec 003
03/09/21
"""

# pip install cvxpy ############################################################
import cvxpy as cp
import numpy as np

def prob1():
    """Solve the following convex optimization problem:

    minimize        2x + y + 3z
    subject to      x  + 2y         <= 3
                         y   - 4z   <= 1
                    2x + 10y + 3z   >= 12
                    x               >= 0
                          y         >= 0
                                z   >= 0

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    # initialize the objective function ########################################
    x = cp.Variable(3, nonneg=True)
    c = np.array([2,1,3])
    objective = cp.Minimize(c.T@x)

    # create constraints #######################################################
    A_0 = np.array([1,2,0])
    A_1 = np.array([0,1,-4])
    A_2 = np.array([2,10,3])
    A_3 = np.eye(3)
    constraints = [A_0@x <= 3, A_1@x <= 1, A_2@x >= 12, A_3@x >= 0]

    # solve optimization problem ###############################################
    problem = cp.Problem(objective, constraints)
    solution = problem.solve()  # must be called for x.value
    return x.value, solution


# Problem 2
def l1Min(A, b):
    """Calculate the solution to the optimization problem

        minimize    ||x||_1
        subject to  Ax = b

    Parameters:
        A ((m,n) ndarray)
        b ((m, ) ndarray)

    Returns:
        The optimizer x (ndarray)
        The optimal value (float)
    """
    # initialize the objective function ########################################
    x = cp.Variable(len(A[0]))
    objective = cp.Minimize(cp.norm(x, 1))

    # create constraints #######################################################
    constraints = [A@x == b]

    # solve optimization problem ###############################################
    problem = cp.Problem(objective, constraints)
    solution = problem.solve()
    return x.value, solution


# Problem 3
def prob3():
    """Solve the transportation problem by converting the last equality constraint
    into inequality constraints.

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    # problem definition #######################################################
    # min 4p_1 + 7p_2 + 6p_3 +8p_4 +8p_5 +9p_6
    # sub to:
    #   A_0: p_1 + p_2 <= 7
    #   A_1: p_3 + p_4 <= 2
    #   A_2: p_5 + p_6 <= 4
    #   A_3: p_1 + p_3 + p_5 == 5
    #   A_4: p_2 + p_4 + p_6 == 8
    #   A_5: p_1, ... , p_6 >= 0
    ############################################################################

    # initialize the objective function ########################################
    x = cp.Variable(6, nonneg=True)
    c = np.array([4,7,6,8,8,9])
    objective = cp.Minimize(c.T @ x)

    # create constraints #######################################################
    A_0 = np.array([1,1,0,0,0,0])
    A_1 = np.array([0,0,1,1,0,0])
    A_2 = np.array([0,0,0,0,1,1])
    A_3 = np.array([1,0,1,0,1,0])
    A_4 = np.array([0,1,0,1,0,1])
    A_5 = np.eye(6)
    constraints = [A_0@x <= 7, A_1@x <=2, A_2@x <=4, A_3@x == 5, A_4@x == 8, A_5@x >=0]

    # solve optimization problem ###############################################
    problem = cp.Problem(objective, constraints)
    solution = problem.solve()
    return x.value, solution


# Problem 4
def prob4():
    """Find the minimizer and minimum of

    g(x,y,z) = (3/2)x^2 + 2xy + xz + 2y^2 + 2yz + (3/2)z^2 + 3x + z

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    # initialize the objective function ########################################
    Q = np.array([[3,2,1],[2,4,2],[1,2,3]])
    r = np.array([3,0,1])
    x = cp.Variable(3)

    # solve optimization problem ###############################################
    problem = cp.Problem(cp.Minimize(.5 * cp.quad_form(x, Q) + r.T@x))
    solution = problem.solve()
    return x.value, solution


# Problem 5
def prob5(A, b):
    """Calculate the solution to the optimization problem
        minimize    ||Ax - b||_2
        subject to  ||x||_1 == 1
                    x >= 0
    Parameters:
        A ((m,n), ndarray)
        b ((m,), ndarray)
        
    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    # initialize the objective function ########################################
    x = cp.Variable(len(A[0]), nonneg=True)
    objective = cp.Minimize(cp.norm((A@x)-b, 2))

    # create constraints #######################################################
    A_0 = np.eye(len(A[0]))
    constraints = [cp.sum(x) == 1, A_0@x >= 0]

    # solve optimization problem ###############################################
    problem = cp.Problem(objective, constraints)
    solution = problem.solve()
    return x.value, solution


# Problem 6
def prob6():
    """Solve the college student food problem. Read the data in the file 
    food.npy to create a convex optimization problem. The first column is 
    the price, second is the number of servings, and the rest contain
    nutritional information. Use cvxpy to find the minimizer and primal 
    objective.
    
    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """	 
    # read data ################################################################
    A = np.load('food.npy', allow_pickle=True)

    # initialize the objective function ########################################
    x = cp.Variable(18, nonneg=True)
    p = A[:,0]  # load price before we multiply per serving
    objective = cp.Minimize(p.T@x)

    # create constraints #######################################################
    for i in range(len(A)):  # calculate nutrition based on servings
        A[i,:] = A[i,1] * A[i,:]

    c = A[:,2]
    f = A[:,3]
    s_hat = A[:,4]
    c_hat = A[:,5]
    f_hat = A[:,6]
    p_hat = A[:,7]
    I = np.eye(18)
    constraints = [c@x <= 2000, f@x <= 65, s_hat@x <= 50, c_hat@x >= 1000, f_hat@x >= 25, p_hat@x >= 46, I@x >= 0]

    # solve optimization problem ###############################################
    problem = cp.Problem(objective, constraints)
    solution = problem.solve()
    return x.value, solution
    
    # The food that should be eaten the most is potatoes. The three best 
    # foods per week are potatoes, milk, and cheese.


#if __name__ == "__main__":
    # test prob1() #############################################################
    # print(prob1())
    ############################################################################


    # test prob2() #############################################################
    #A = np.array([[1,2,1,1],[0,3,-2,-1]])
    #b = np.array([7,4])
    #print(l1Min(A, b))
    ############################################################################


    # test prob3() #############################################################
    #print(prob3())
    ############################################################################


    # test prob4() #############################################################
    #print(prob4())
    ############################################################################


    # test prob5() #############################################################
    #A = np.array([[1,2,1,1],[0,3,-2,-1]])
    #b = np.array([7,4])
    #print(prob5(A, b))
    ############################################################################


    # test prob6() #############################################################
    #print(prob6())
    ############################################################################