# nearest_neighbor.py
"""Volume 2: Nearest Neighbor Search.
Marcelo Leszynski
Math 321 Sec 005
10/22/20
"""

import numpy as np
from scipy import linalg as la
from scipy.spatial import KDTree
import scipy.stats
from matplotlib import pyplot as plt


# Problem 1
def exhaustive_search(X, z):
    """Solve the nearest neighbor search problem with an exhaustive search.

    Parameters:
        X ((m,k) ndarray): a training set of m k-dimensional points.
        z ((k, ) ndarray): a k-dimensional target point.

    Returns:
        ((k,) ndarray) the element (row) of X that is nearest to z.
        (float) The Euclidean distance from the nearest neighbor to z.
    """
    # calculate the distance
    x_z = la.norm(X-z, axis = 1)
    # find the index of the minimum distance
    index = np.argmin(x_z)

    return X[index], min(la.norm(X-z, axis=1))


# Problem 2: Write a KDTNode class.
class KDTNode:
    """Node class for K-D Trees.

    Attributes:
        left (KDTNode): a reference to this node's left child.
        right (KDTNode): a reference to this node's right child.
        value ((k,) ndarray): a coordinate in k-dimensional space.
        pivot (int): the dimension of the value to make comparisons on.
    """
    def __init__(self, x):
        if type(x) is not np.ndarray:
            raise TypeError(str(x) + " is not a numpy array.")
        self.value = x
        self.left = None
        self.right = None
        self.pivot = None

# Problems 3 and 4
class KDT:
    """A k-dimensional binary tree for solving the nearest neighbor problem.

    Attributes:
        root (KDTNode): the root node of the tree. Like all other nodes in
            the tree, the root has a NumPy array of shape (k,) as its value.
        k (int): the dimension of the data in the tree.
    """
    def __init__(self):
        """Initialize the root and k attributes."""
        self.root = None
        self.k = None

    def find(self, data):
        """Return the node containing the data. If there is no such node in
        the tree, or if the tree is empty, raise a ValueError.
        """
        def _step(current):
            """Recursively step through the tree until finding the node
            containing the data. If there is no such node, raise a ValueError.
            """
            if current is None:                     # Base case 1: dead end.
                raise ValueError(str(data) + " is not in the tree")
            elif np.allclose(data, current.value):
                return current                      # Base case 2: data found!
            elif data[current.pivot] < current.value[current.pivot]:
                return _step(current.left)          # Recursively search left.
            else:
                return _step(current.right)         # Recursively search right.

        # Start the recursive search at the root of the tree.
        return _step(self.root)

    # Problem 3
    def insert(self, data):
        """Insert a new node containing the specified data.

        Parameters:
            data ((k,) ndarray): a k-dimensional point to insert into the tree.

        Raises:
            ValueError: if data does not have the same dimensions as other
                values in the tree.
            ValueError: if data is already in the tree
        """
        new_node = KDTNode(data)
        current = self.root

        # recursively find location for new node ###############################
        def _step(current):
            # insert into an empty tree ########################################
            if self.root is None:
                self.root = new_node
                self.root.pivot = 0
                self.k = len(data)

            else:
                if data.size != self.k:  # check dimensions
                    raise ValueError(str(data) + " is not in the R^k")
                if np.allclose(data, current.value):  # check for duplicates
                    raise ValueError(str(data) + " is already in the tree")
                # recursively insert left ##################################
                elif data[current.pivot] < current.value[current.pivot]:
                    if current.left is None:
                        current.left = new_node
                        current.left.pivot = (current.pivot + 1) % self.k
                    else:
                        return _step(current.left)
                # recursively insert right #################################
                else:
                    if current.right is None:
                        current.right = new_node
                        current.right.pivot = (current.pivot + 1) % self.k
                    else:
                        return _step(current.right)

        return _step(current)


    # Problem 4
    def query(self, z):
        """Find the value in the tree that is nearest to z.

        Parameters:
            z ((k,) ndarray): a k-dimensional target point.

        Returns:
            ((k,) ndarray) the value in the tree that is nearest to z.
            (float) The Euclidean distance from the nearest neighbor to z.
        """
        # implementing the given algorithm
        def KDSearch(current, nearest, d):
            if current is None:
                return nearest, d
            x = current.value
            i = current.pivot
            if la.norm(x-z) < d:
                nearest = current
                d = la.norm(x-z)
            if z[i] < x[i]:
                nearest, d = KDSearch(current.left, nearest, d)
                if z[i] + d >= x[i]:
                    nearest, d = KDSearch(current.right, nearest, d)
            else:
                nearest, d = KDSearch(current.right, nearest, d)
                if z[i] - d <= x[i]:
                    nearest, d = KDSearch(current.left, nearest, d)
            return nearest, d

        node, d = KDSearch(self.root, self.root, la.norm(self.root.value -z))
        return node.value, d

    def __str__(self):
        """String representation: a hierarchical list of nodes and their axes.

        Example:                           'KDT(k=2)
                    [5,5]                   [5 5]   pivot = 0
                    /   \                   [3 2]   pivot = 1
                [3,2]   [8,4]               [8 4]   pivot = 1
                    \       \               [2 6]   pivot = 0
                    [2,6]   [7,5]           [7 5]   pivot = 0'
        """
        if self.root is None:
            return "Empty KDT"
        nodes, strs = [self.root], []
        while nodes:
            current = nodes.pop(0)
            strs.append("{}\tpivot = {}".format(current.value, current.pivot))
            for child in [current.left, current.right]:
                if child:
                    nodes.append(child)
        return "KDT(k={})\n".format(self.k) + "\n".join(strs)


# Problem 5: Write a KNeighborsClassifier class.
class KNeighborsClassifier:
    """A k-nearest neighbors classifier that uses SciPy's KDTree to solve
    the nearest neighbor problem efficiently.
    """
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors
        self.tree = None
        self.lable = None

    # create a tree and assign the label
    def fit(self, X, y):
        tree = KDTree(X)
        self.tree = tree
        self.label = None
    
    # predict the z
    def predict(self, z):
        d, index = self.tree.query(z, k = self.n_neighbors)
        return scipy.stats.mode(self.label[index])[0][0]


# Problem 6
def prob6(n_neighbors, filename="mnist_subset.npz"):
    """Extract the data from the given file. Load a KNeighborsClassifier with
    the training data and the corresponding labels. Use the classifier to
    predict labels for the test data. Return the classification accuracy, the
    percentage of predictions that match the test labels.

    Parameters:
        n_neighbors (int): the number of neighbors to use for classification.
        filename (str): the name of the data file. Should be an npz file with
            keys 'X_train', 'y_train', 'X_test', and 'y_test'.

    Returns:
        (float): the classification accuracy.
    """
    # extract the data and initialize variables ################################
    data = np.load("mnist_subset.npz")
    X_train = data["X_train"].astype(np.float)  # trainig data
    y_train = data["y_train"] # training labels
    X_test = data["X_test"].astype(np.float)  # test data
    y_test = data["y_test"]

    # train the classifier with the training set ###############################
    image = KNeighborsClassifier(n_neighbors)
    image.fit(X_train, y_train)

    accuracy = 0

    for i in range(0, len(X_test)):
        if image.predict(X_test[i] == y_test[i]):
            accuracy += 1
    
    # plot the graph ###########################################################
    plt.imshow(X_test[0].reshape((28,28)), cmap="gray")
    plt.show()

    return accuracy / len(y_test)