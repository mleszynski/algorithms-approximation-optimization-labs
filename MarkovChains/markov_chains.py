# markov_chains.py
"""Volume 2: Markov Chains.
Marcelo Leszynski
Math 321 sec 005
11/7/20
"""

import numpy as np
import math
from scipy import linalg as la


class MarkovChain:
    """A Markov chain with finitely many states.

    Attributes:
        _A ((n,n) ndarray): a column_stochastic transition matrix for a 
            Markov chain with n states.
        _labels (list(str)): a list of labels for the columns of the  
            transition matrix.
        _dictionary (dict(str:int)): a dictionary that correlates a label to 
            the corresponding column in the transition matrix.
        _n (int): an integer to keep track of the dimensions of the transition 
            matrix.
    """
    # Problem 1
    def __init__(self, A, states=None):
        """Check that A is column stochastic and construct a dictionary
        mapping a state's label to its index (the row / column of A that the
        state corresponds to). Save the transition matrix, the list of state
        labels, and the label-to-index dictionary as attributes.

        Parameters:
        A ((n,n) ndarray): the column-stochastic transition matrix for a
            Markov chain with n states.
        states (list(str)): a list of n labels corresponding to the n states.
            If not provided, the labels are the indices 0, 1, ..., n-1.

        Raises:
            ValueError: if A is not square or is not column stochastic.

        Example:
            >>> MarkovChain(np.array([[.5, .8], [.5, .2]], states=["A", "B"])
        corresponds to the Markov Chain with transition matrix
                                   from A  from B
                            to A [   .5      .8   ]
                            to B [   .5      .2   ]
        and the label-to-index dictionary is {"A":0, "B":1}.
        """
        # check for valid input ################################################
        num_rows, num_cols = np.shape(A)
        if num_rows != num_cols:
            raise ValueError("Given matrix is not square")
        #if not math.isclose(total, 1.):
        if not np.allclose(np.ones(num_cols), A.sum(axis=0)):
            raise ValueError("Given matrix is not column stochastic")

        # create and store labels ##############################################
        if states is None:
            self._labels = list(range(num_rows))
        else:
            self._labels = states

        # create and store other attributes ####################################
        self._dictionary = {self._labels[i]: i for i in range(len(self._labels))}
        self._A = A
        self._n = num_rows
        
        
    # Problem 2
    def transition(self, state):
        """Transition to a new state by making a random draw from the outgoing
        probabilities of the state with the specified label.

        Parameters:
            state (str): the label for the current state.

        Returns:
            (str): the label of the state to transitioned to.
        """
        index = self._dictionary[state]  # turn the given state into a transition matrix index
        results = np.random.multinomial(1, self._A[:,index])  # calculate the distribution at the index
        return self._labels[np.argmax(results)]  # return the label corresponding to distribution results

    # Problem 3
    def walk(self, start, N):
        """Starting at the specified state, use the transition() method to
        transition from state to state N-1 times, recording the state label at
        each step.

        Parameters:
            start (str): The starting state label.

        Returns:
            (list(str)): A list of N state labels, including start.
        """
        # initialize necessary vars ############################################
        output = [start]
        current_state = start

        # use self.transition to transition N-1 times ##########################
        for i in range(N-1):
            current_state = self.transition(current_state)
            output.append(current_state)

        return output

    # Problem 3
    def path(self, start, stop):
        """Beginning at the start state, transition from state to state until
        arriving at the stop state, recording the state label at each step.

        Parameters:
            start (str): The starting state label.
            stop (str): The stopping state label.

        Returns:
            (list(str)): A list of state labels from start to stop.
        """
        # initialize necessary vars ############################################
        output = [start]
        current_state = start

        # use self.transition to transition until stop-state ###################
        while current_state != stop:
            current_state = self.transition(current_state)
            output.append(current_state)
        
        return output

    # Problem 4
    def steady_state(self, tol=1e-12, maxiter=40):
        """Compute the steady state of the transition matrix A.

        Parameters:
            tol (float): The convergence tolerance.
            maxiter (int): The maximum number of iterations to compute.

        Returns:
            ((n,) ndarray): The steady state distribution vector of A.

        Raises:
            ValueError: if there is no convergence within maxiter iterations.
        """
        # generate a random state distribution #################################
        rand_state = np.random.random((self._n, 1))
        denom = np.sum(rand_state)
        rand_state = rand_state / denom
        prev_state = np.copy(rand_state)

        # calculate product Ax #################################################
        for i in range(maxiter):
            prev_state = np.copy(rand_state)
            rand_state = self._A @ rand_state

            # use the 1-norm to check if arrived at steady state ###############
            if np.linalg.norm(rand_state-prev_state, ord=1) < tol:
                return rand_state

        # if no steady state is found after maxiter iterations #################
        raise ValueError("A^k does not converge in " + str(maxiter) + " iterations")

class SentenceGenerator(MarkovChain):
    """A Markov-based simulator for natural language.

    Attributes:
        _A ((n,n) ndarray): a column_stochastic transition matrix for a 
            Markov chain with n states.
        _labels (list(str)): a list of labels for the columns of the  
            transition matrix.
        _dictionary (dict(str:int)): a dictionary that correlates a label to 
            the corresponding column in the transition matrix.
        _n (int): an integer to keep track of the dimensions of the transition 
            matrix.
    """
    # Problem 5
    def __init__(self, filename):
        """Read the specified file and build a transition matrix from its
        contents. You may assume that the file has one complete sentence
        written on each line.
        """
        # initialize necessary vars ############################################
        words = None
        sentences = None

        # read in training file contents #######################################
        with open(filename, 'r') as my_file:

            contents = my_file.read()
            # store words and sentences from training file, excluding trailing '\n'
            words = contents.split()
            sentences = contents.split('\n')
            sentences.pop()

        # remove duplicates and add "$tart" and "$top" to our words list #######
        words = set(words)
        words = list(words)
        words.insert(0, "$tart")
        words.append("$top")

        # initialize a zero'd out transition matrix and index dictionary #######
        A = np.zeros((len(words),len(words)))
        dictionary = {words[i]: i for i in range(len(words))}

        # use sentences to fill out transition matrix ##########################
        for sentence in sentences:
            # prep sentence for processing into transition matrix ##############
            temp_words = sentence.split()
            temp_words.insert(0, "$tart")
            temp_words.append("$top")

            # iterate across word pairs and insert connections in transition matrix
            for i in range(len(temp_words) - 1):
                A[dictionary[temp_words[i+1]], dictionary[temp_words[i]]] = A[dictionary[temp_words[i+1]], dictionary[temp_words[i]]] + 1
        
        # final prep to make transition matrix stochastic ######################
        A[dictionary["$top"],dictionary["$top"]] = 1
       
        for i in range(len(words)):  # normalizing columns
            denominator = np.sum(A[:,i], axis=0)
            A[:,i] = A[:,i] / denominator

        # constructing other attributes
        self._A = A
        self._dictionary = dictionary
        self._labels = words
        rows, cols = np.shape(A)
        self._n = rows

    # Problem 6
    def babble(self):
        """Create a random sentence using MarkovChain.path().

        Returns:
            (str): A sentence generated with the transition matrix, not
                including the labels for the $tart and $top states.

        Example:
            >>> yoda = SentenceGenerator("yoda.txt")
            >>> print(yoda.babble())
            The dark side of loss is a path as one with you.
        """
        path = MarkovChain.path(self, "$tart", "$top")  # get the path
        path = path[1:-1]  # remove "$tart" and "$top"
        output = ' '
        return output.join(path)