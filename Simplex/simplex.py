"""Volume 2: Simplex

Marcelo Leszynski
Math 323 Sec 003
03/04/21
"""

import numpy as np


# Problems 1-6
class SimplexSolver(object):
    """Class for solving the standard linear optimization problem

                        minimize        c^Tx
                        subject to      Ax <= b
                                         x >= 0
    via the Simplex algorithm.
    """
    # Problem 1
    def __init__(self, c, A, b):
        """Check for feasibility and initialize the dictionary.

        Parameters:
            c ((n,) ndarray): The coefficients of the objective function.
            A ((m,n) ndarray): The constraint coefficients matrix.
            b ((m,) ndarray): The constraint vector.

        Raises:
            ValueError: if the given system is infeasible at the origin.
        """
        # test for problem feasibility #########################################
        test_vect = A @ np.zeros(len(A[0]))
        test_vect = test_vect <= b
        if np.sum(test_vect) != len(A):
            raise ValueError("Problem is not feasible at origin.")

        # initialize object members ############################################
        self._c = c
        self._A = A
        self._b = b
        self._D = None 


    # Problem 2
    def _generatedictionary(self, c, A, b):
        """Generate the initial dictionary.

        Parameters:
            c ((n,) ndarray): The coefficients of the objective function.
            A ((m,n) ndarray): The constraint coefficients matrix.
            b ((m,) ndarray): The constraint vector.
        """
        # create c_bar vector ##################################################
        c_zeros = np.zeros(len(A))
        c_bar = np.concatenate(([0], c))
        c_bar = np.concatenate((c_bar, c_zeros))

        # create A_bar matrix ##################################################
        I_m = np.identity(len(A))
        A_bar = -np.concatenate((A, I_m), axis=1)
        A_bar = np.concatenate((b.reshape(-1,1), A_bar), axis=1)

        # create dictionary by joining vectors and matrices ####################
        self._D = np.vstack((c_bar, A_bar))


    # Problem 3a
    def _pivot_col(self):
        """Return the column index of the next pivot column.
        """
        # case where valid index is found ######################################
        for i in range(1, len(self._D[0])):
            if self._D[0,i] < 0:
                return i

        # simplex terminating case #############################################
        return 0

    # Problem 3b
    def _pivot_row(self, index):
        """Determine the row index of the next pivot row using the ratio test
        (Bland's Rule).
        """
        # check for valid input index ##########################################
        if index == 0:
            return 0

        # check for unboundedness ##############################################
        column = self._D[:,index]
        mask = column < np.zeros(len(column))

        if np.sum(mask) == 0:
            return 0

        # return minimal index for pivot #######################################
        ratios = np.zeros(len(mask))
        b_vect = self._D[:,0]
        for i in range(1, len(ratios)):
            if mask[i]:
                ratios[i] = -b_vect[i] / column[i]

        for i in range(1, len(ratios)):
            if not mask[i]:
                ratios[i] = np.nan

        return np.nanargmin(ratios[1:]) + 1


    # Problem 4
    def pivot(self):
        """Select the column and row to pivot on. Reduce the column to a
        negative elementary vector.
        """
        # get the indices for the pivot row and column #########################
        col_index = self._pivot_col()
        row_index = self._pivot_row(col_index)

        # check for boundedness ################################################
        if col_index == 0 or row_index == 0:
            raise ValueError("Problem is unbounded.")

        # divide the pivot row #################################################
        self._D[row_index,:] = self._D[row_index,:] / -self._D[row_index,col_index]

        # use the pivot row to zero out other values ###########################
        for i in range(len(self._D)):
            # don't make changes to the pivot row ##############################
            if i == row_index:
                continue

            # use pivot row to change other rows ###############################
            scalar = self._D[i, col_index] / self._D[row_index, col_index]
            self._D[i,:] = self._D[i,:] - scalar * self._D[row_index,:]



    # Problem 5
    def solve(self):
        """Solve the linear optimization problem.

        Returns:
            (float) The minimum value of the objective function.
            (dict): The basic variables and their values.
            (dict): The nonbasic variables and their values.
        """
        # generate the necessary dictionary ####################################
        self._generatedictionary(self._c, self._A, self._b)

        # check for terminating condition ######################################
        num_nonneg = np.sum(self._D[0,1:] >= 0)
        while num_nonneg != (len(self._D[0]) - 1):
            self.pivot()
            num_nonneg = np.sum(self._D[0,1:] >= 0)

        # create indices necessary for return values ###########################
        is_dependent = self._D[0,1:] == 0
        ind_indices = []
        dep_indices = []
        for i in range(len(is_dependent)):
            if is_dependent[i]:
                dep_indices.append(i)
            else:
                ind_indices.append(i)
        
        dep_values = []
        for index in dep_indices:
            dep_values.append(self._D[np.argmin(self._D[:,index+1]),0])

        return self._D[0,0], {dep_indices[i]:dep_values[i] for i in range(len(dep_indices))}, {ind_indices[i]:0 for i in range(len(ind_indices))}


# Problem 6
def prob6(filename='productMix.npz'):
    """Solve the product mix problem for the data in 'productMix.npz'.

    Parameters:
        filename (str): the path to the data file.

    Returns:
        ((n,) ndarray): the number of units that should be produced for each product.
    """
    # load in data from file ###################################################
    data = np.load(filename)
    data_A = data['A']
    c = -data['p']
    data_m = data['m']
    data_d = data['d']

    # create and solve the system ##############################################
    A = np.vstack((data_A,np.identity(len(data_A[0]))))
    b = np.concatenate((data_m, data_d))
    solver = SimplexSolver(c, A, b)
    sol = solver.solve()

    # return wanted info #######################################################
    dictionary = sol[1]
    return [dictionary[i] for i in range(4)]

    

    

if __name__ == "__main__":
    # test prob1() #############################################################
    #c_vect = np.array([-3,-2])
    #A = np.array([[1,-1],[3,1],[4,3]])
    #b_vect = np.array([2,5,7])
    ############################################################################


    # test prob2() #############################################################
    #c_vect = np.array([-3,-2])
    #A = np.array([[1,-1],[3,1],[4,3]])
    #b_vect = np.array([2,5,7])
    #solver = SimplexSolver(c_vect, A, b_vect)
    #solver._generatedictionary(c_vect, A, b_vect)
    #print('printing dictionary')
    #print(solver._D)
    ############################################################################


    # test prob3() and prob4() #################################################
    #c_vect = np.array([-3,-2])
    #A = np.array([[1,-1],[3,1],[4,3]])
    #b_vect = np.array([2,5,7])
    #solver = SimplexSolver(c_vect, A, b_vect)
    #solver.pivot()
    #print('printing pivoted dictionary')
    #print(solver._D)
    ############################################################################


    # test prob5() #############################################################
    #c_vect = np.array([-3,-2])
    #A = np.array([[1,-1],[3,1],[4,3]])
    #b_vect = np.array([2,5,7])
    #solver = SimplexSolver(c_vect, A, b_vect)
    #sol = solver.solve()
    #print('printing solution')
    #print(sol)
    ############################################################################


    # test prob6() #############################################################
    print(prob6())
    ############################################################################