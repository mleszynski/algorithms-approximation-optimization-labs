# dynamic_programming.py
"""Volume 2: Dynamic Programming.
Marcelo Leszynski
Math 323 Sec 003
04/01/20
"""

import numpy as np
from matplotlib import pyplot as plt


# Problem 1
def calc_stopping(N):
    """Calculate the optimal stopping time and expected value for the
    marriage problem.

    Parameters:
        N (int): The number of candidates.

    Returns:
        (float): The maximum expected value of choosing the best candidate.
        (int): The index of the maximum expected value.
    """
    # create base case #########################################################
    probs_list = [0]

    # calculate rest of expected values iteratively ############################
    for i in range(N):
        probs_list.append(max((N-i-1)/(N-i)*probs_list[-1] + (1/N), probs_list[-1]))

    # find the optimal stopping point ##########################################
        if probs_list[-1] == probs_list[-2]:
            return probs_list[-1], N-i-1


# Problem 2
def graph_stopping_times(M):
    """Graph the optimal stopping percentage of candidates to interview and
    the maximum probability against M.

    Parameters:
        M (int): The maximum number of candidates.

    Returns:
        (float): The optimal stopping percent of candidates for M.
    """
    # initialize lists #########################################################
    max_probs = []
    percentages = []

    # calculate values for different N #########################################
    for N in range(3, M+1):
        data = calc_stopping(N)
        max_probs.append(data[0])
        percentages.append(data[1]/N)

    # plot values ##############################################################
    domain = range(3, M+1)
    plt.plot(domain, max_probs, '-r', label='Maximum Probabilities')
    plt.plot(domain, percentages, '-b', label='Stopping Percentages')
    plt.title('Problem 2')
    plt.legend(loc='upper right')
    plt.show()

    return percentages[-1]


# Problem 3
def get_consumption(N, u=lambda x: np.sqrt(x)):
    """Create the consumption matrix for the given parameters.

    Parameters:
        N (int): Number of pieces given, where each piece of cake is the
            same size.
        u (function): Utility function.

    Returns:
        C ((N+1,N+1) ndarray): The consumption matrix.
    """
    # initialize values ########################################################
    c_matrix = np.zeros((N+1, N+1))
    w_vect = np.linspace(0,1,N+1)

    # create utility matrix ####################################################
    for i in range(len(c_matrix)):
        for j in range(len(c_matrix[0])):
            if j <= i:
                c_matrix[i,j] = u(w_vect[i]-w_vect[j])

    return c_matrix


# Problems 4-6
def eat_cake(T, N, B, u=lambda x: np.sqrt(x)):
    """Create the value and policy matrices for the given parameters.

    Parameters:
        T (int): Time at which to end (T+1 intervals).
        N (int): Number of pieces given, where each piece of cake is the
            same size.
        B (float): Discount factor, where 0 < B < 1.
        u (function): Utility function.

    Returns:
        A ((N+1,T+1) ndarray): The matrix where the (ij)th entry is the
            value of having w_i cake at time j.
        P ((N+1,T+1) ndarray): The matrix where the (ij)th entry is the
            number of pieces to consume given i pieces at time j.
    """
    # initialize values ########################################################
    A = np.zeros((N+1, T+1))
    P = np.zeros((N+1, T+1))
    w_vect = np.linspace(0,1,N+1)

    # calculate initial value and policy matrices ##############################
    for i in range(len(A)):
        A[i,-1] = u(w_vect[i])
        P[i,-1] = w_vect[i]

    # for col in A and P #######################################################
    for col in range(1, len(A[0])):
        # calculate current value matrix #######################################
        CV = np.zeros((N+1, N+1))
        for i in range(len(CV)):
            for j in range(len(CV)):
                if j<= i:
                    CV[i,j] = u(w_vect[i]-w_vect[j]) + B*A[j, -col]
        
        # for row in A and P ###################################################
        for row in range(len(A)):
            A[row, -col-1] = np.max(CV[row])
            P[row, -col-1] = w_vect[row]-w_vect[np.argmax(CV[row])]

    return A, P


# Problem 7
def find_policy(T, N, B, u=np.sqrt):
    """Find the most optimal route to take assuming that we start with all of
    the pieces. Show a graph of the optimal policy using graph_policy().

    Parameters:
        T (int): Time at which to end (T+1 intervals).
        N (int): Number of pieces given, where each piece of cake is the
            same size.
        B (float): Discount factor, where 0 < B < 1.
        u (function): Utility function.

    Returns:
        ((T,) ndarray): The matrix describing the optimal percentage to
            consume at each time.
    """
    # get policy matrix and initialize policy vector ###########################
    A, P = eat_cake(T, N, B, u)
    c = np.zeros(len(P[0]))
    c[0] = P[-1,0]

    # calculate the rest of the optimal policy #################################
    for i in range(1, len(c)):
        c[i] = P[int(N-(np.sum(c)*N)),i]

    return c


#if __name__ == "__main__":
    # test prob 1 ##############################################################
    #print(calc_stopping(4))
    ############################################################################


    # test prob 2 ##############################################################
    #print(graph_stopping_times(1000))
    ############################################################################


    # test prob 3 ##############################################################
    #print(get_consumption(4))
    ############################################################################


    # test prob 4-6 ############################################################
    #print(eat_cake(3, 4, 0.9))
    ############################################################################


    # test prob 7 ##############################################################
    #print(find_policy(3, 4, 0.9))
    ############################################################################