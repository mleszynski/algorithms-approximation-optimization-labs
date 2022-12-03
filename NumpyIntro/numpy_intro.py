# numpy_intro.py
"""Python Essentials: Intro to NumPy.
Marcelo Leszynski
MATH 321 Sec 005
09/08/20
"""
import numpy as np


def prob1():
    """Define the matrices A and B as arrays. Return the matrix product AB."""
    A = np.array([[3, -1, 4], [1, 5, -9]])
    B = np.array([[2, 6, -5, 3], [5, -8, 9, 7], [9, -3, -2, -3]])
    return np.dot(A, B)
    raise NotImplementedError("Problem 1 Incomplete")


def prob2():
    """Define the matrix A as an array. Return the matrix -A^3 + 9A^2 - 15A."""
    A = np.array([[3, 1, 4], [1, 5, 9], [-5, 3, 1]])
    return (-1 * np.dot(np.dot(A, A), A)) + (9 * np.dot(A, A)) - (15 * A)
    raise NotImplementedError("Problem 2 Incomplete")


def prob3():
    """Define the matrices A and B as arrays. Calculate the matrix product ABA,
    change its data type to np.int64, and return it.
    """
    A = np.triu(np.ones((7, 7)))
    B = np.tril(np.full((7, 7), -1)) + (np.triu(np.full((7, 7), 5) - np.diag([5, 5, 5, 5, 5, 5, 5])))
    ABA = np.dot(A, np.dot(B, A))
    ABA = ABA.astype(np.int64)
    return ABA
    raise NotImplementedError("Problem 3 Incomplete")


def prob4(A):
    """Make a copy of 'A' and set all negative entries of the copy to 0.
    Return the copy.

    Example:
        >>> A = np.array([-3,-1,3])
        >>> prob4(A)
        array([0, 0, 3])
    """
    temp = np.copy(A)
    mask = temp < 0
    temp[mask] = 0
    return temp
    raise NotImplementedError("Problem 4 Incomplete")


def prob5():
    """Define the matrices A, B, and C as arrays. Return the block matrix
                                | 0 A^T I |
                                | A  0  0 |,
                                | B  0  C |
    where I is the 3x3 identity matrix and each 0 is a matrix of all zeros
    of the appropriate size.
    """
    A = np.array([[0, 2, 4], [1, 3, 5]])
    B = np.tril(np.full((3, 3),  3))
    C = np.diag([-2, -2, -2])
    col1 = np.vstack((np.zeros((3, 3)), A, B))
    col2 = np.vstack((A.T, np.zeros((5, 2))))
    col3 = np.vstack((np.eye(3), np.zeros((2, 3)), C))
    final = np.hstack((col1, col2, col3))
    return final
    raise NotImplementedError("Problem 5 Incomplete")


def prob6(A):
    """Divide each row of 'A' by the row sum and return the resulting array.

    Example:
        >>> A = np.array([[1,1,0],[0,1,0],[1,1,1]])
        >>> prob6(A)
        array([[ 0.5       ,  0.5       ,  0.        ],
               [ 0.        ,  1.        ,  0.        ],
               [ 0.33333333,  0.33333333,  0.33333333]])
    """
    temp = np.copy(A)
    divisor = temp.sum(axis = 1).reshape((-1, 1))
    return temp / divisor
    raise NotImplementedError("Problem 6 Incomplete")


def prob7():
    """Given the array stored in grid.npy, return the greatest product of four
    adjacent numbers in the same direction (up, down, left, right, or
    diagonally) in the grid.
    """
    grid = np.load("grid.npy")
    rowsum = np.max(grid[:, :-3] * grid[:, 1:-2] * grid[:, 2:-1] * grid[:, 3:])
    colsum = np.max(grid[:-3, :] * grid[1:-2, :] * grid[2:-1, :] * grid[3:, :])
    diagsum1 = np.max(grid[:-3, :-3] * grid[1:-2, 1:-2] * grid[2:-1, 2:-1] * grid[3:, 3:])
    diagsum2 = np.max(grid[:-3, 3:] * grid[1:-2, 2:-1] * grid[2:-1, 1:-2] * grid[3:, :-3])
    return max(rowsum, colsum, diagsum1, diagsum2)
    raise NotImplementedError("Problem 7 Incomplete")
