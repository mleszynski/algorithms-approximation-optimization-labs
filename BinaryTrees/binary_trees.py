# binary_trees.py
"""Volume 2: Binary Trees.
Marcelo Leszynski
Math 321 Sec 001
10/20/20
"""

import time as time
import random as rand
from matplotlib import pyplot as plt
# These imports are used in BST.draw().
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

class SinglyLinkedListNode:
    """A node with a value and a reference to the next node."""
    def __init__(self, data):
        self.value, self.next = data, None

class SinglyLinkedList:
    """A singly linked list with a head and a tail."""
    def __init__(self):
        self.head, self.tail = None, None

    def append(self, data):
        """Add a node containing the data to the end of the list."""
        n = SinglyLinkedListNode(data)
        if self.head is None:
            self.head, self.tail = n, n
        else:
            self.tail.next = n
            self.tail = n

    def iterative_find(self, data):
        """Search iteratively for a node containing the data.
        If there is no such node in the list, including if the list is empty,
        raise a ValueError.

        Returns:
            (SinglyLinkedListNode): the node containing the data.
        """
        current = self.head
        while current is not None:
            if current.value == data:
                return current
            current = current.next
        raise ValueError(str(data) + " is not in the list")

    # Problem 1
    def recursive_find(self, data):
        """Search recursively for the node containing the data.
        If there is no such node in the list, including if the list is empty,
        raise a ValueError.

        Returns:
            (SinglyLinkedListNode): the node containing the data.
        """
        # define a recursive function to traverse the list #####################
        def _step(current):
            # base case 1: dead end ############################################
            if current is None:                  
                raise ValueError(str(data) + " is not in the list")
            # base case 2: data found ##########################################
            if data == current.value:  
                return current
            # recursive step ###################################################
            else:
                return _step(current.next)
    
        # start the recursion on the head of the list ##########################
        return _step(self.head)


class BSTNode:
    """A node class for binary search trees. Contains a value, a
    reference to the parent node, and references to two child nodes.
    """
    def __init__(self, data):
        """Construct a new node and set the value attribute. The other
        attributes will be set when the node is added to a tree.
        """
        self.value = data
        self.prev = None        # A reference to this node's parent node.
        self.left = None        # self.left.value < self.value
        self.right = None       # self.value < self.right.value


class BST:
    """Binary search tree data structure class.
    The root attribute references the first node in the tree.
    """
    def __init__(self):
        """Initialize the root attribute."""
        self.root = None

    def find(self, data):
        """Return the node containing the data. If there is no such node
        in the tree, including if the tree is empty, raise a ValueError.
        """

        # Define a recursive function to traverse the tree.
        def _step(current):
            """Recursively step through the tree until the node containing
            the data is found. If there is no such node, raise a Value Error.
            """
            if current is None:                     # Base case 1: dead end.
                raise ValueError(str(data) + " is not in the tree.")
            if data == current.value:               # Base case 2: data found!
                return current
            if data < current.value:                # Recursively search left.
                return _step(current.left)
            else:                                   # Recursively search right.
                return _step(current.right)

        # Start the recursion on the root of the tree.
        return _step(self.root)

    # Problem 2
    def insert(self, data):
        """Insert a new node containing the specified data.

        Raises:
            ValueError: if the data is already in the tree.

        Example:
            >>> tree = BST()                    |
            >>> for i in [4, 3, 6, 5, 7, 8, 1]: |            (4)
            ...     tree.insert(i)              |            / \
            ...                                 |          (3) (6)
            >>> print(tree)                     |          /   / \
            [4]                                 |        (1) (5) (7)
            [3, 6]                              |                  \
            [1, 5, 7]                           |                  (8)
            [8]                                 |
        """
        # define a recursive function to find the insertion parent node ########
        def _pstep(current):
            """Finds the parent node location for the insertion of a
            new node.

            Raises:
                ValueError: if the data is already in the tree.
            """
            # base case 1: already exists ######################################
            if data == current.value:
                raise ValueError(str(data) + " is already in the BST")
            # recursively search for correct parent node #######################
            if data > current.value:
                if current.right is None:  # base case
                    return current
                else:
                    return _pstep(current.right)  # recursively search right child
            else:
                if current.left is None:  # base case
                    return current
                else:
                    return _pstep(current.left)  # recursively search left child
        
        # create a new node ####################################################
        new_node = BSTNode(data)

        # insert the node in the correct location ##############################
        if self.root is None:  # if tree is empty, add to root
            self.root = new_node
        else:
            parent_node = _pstep(self.root)
            new_node.prev = parent_node
            if new_node.value > parent_node.value:  # insert right
                parent_node.right = new_node
            else:  # insert left
                parent_node.left = new_node

    # Problem 3
    def remove(self, data):
        """Remove the node containing the specified data.

        Raises:
            ValueError: if there is no node containing the data, including if
                the tree is empty.

        Examples:
            >>> print(12)                       | >>> print(t3)
            [6]                                 | [5]
            [4, 8]                              | [3, 6]
            [1, 5, 7, 10]                       | [1, 4, 7]
            [3, 9]                              | [8]
            >>> for x in [7, 10, 1, 4, 3]:      | >>> for x in [8, 6, 3, 5]:
            ...     t1.remove(x)                | ...     t3.remove(x)
            ...                                 | ...
            >>> print(t1)                       | >>> print(t3)
            [6]                                 | [4]
            [5, 8]                              | [1, 7]
            [9]                                 |
                                                | >>> print(t4)
            >>> print(t2)                       | [5]
            [2]                                 | >>> t4.remove(1)
            [1, 3]                              | ValueError: <message>
            >>> for x in [2, 1, 3]:             | >>> t4.remove(5)
            ...     t2.remove(x)                | >>> print(t4)
            ...                                 | []
            >>> print(t2)                       | >>> t4.remove(5)
            []                                  | ValueError: <message>
        """
        def getImP(target):
            """Returns the immediate predecessor of a node by 
            passing in the left child as the target.
            """
            # base case ########################################################
            if target.right is None:
                return target
            # recursive step ###################################################
            else:
                return getImP(target.right)


        # handle the case where the tree is empty ##############################
        if self.root is None:  
            raise ValueError("Tree is empty")
        # find node to remove ##################################################
        target_node = self.find(data)
        # handle the case where the target node is the root ####################
        if self.root is target_node:
            # remove root - no children case ###################################
            if self.root.left is None and self.root.right is None:
                self.root = None
            # remove root - one child cases ####################################
            elif self.root.left is None and self.root.right is not None:
                self.root = self.root.right
                self.root.prev = None
            elif self.root.left is not None and self.root.right is None:
                self.root = self.root.left
                self.root.prev = None
            # remove root - two child case #####################################
            else:
                temp_val = getImP(self.root.left).value
                self.remove(temp_val)
                self.root.value = temp_val
        # handle non-root removals #############################################
        else:
            # find location of target_node relative to parent node #############
            is_target_left = True
            if target_node.prev.right is target_node:
                is_target_left = False
            # remove leaf node #################################################
            if target_node.left is None and target_node.right is None:
                # leaf is to the left of parent ################################
                if is_target_left:
                    target_node.prev.left = None
                # leaf is to the right of parent################################
                else:
                    target_node.prev.right = None
            # remove non-root - one child cases ################################ 
            elif target_node.left is not None and target_node.right is None:
                if is_target_left:
                    target_node.prev.left = target_node.left
                else:
                    target_node.prev.right = target_node.left
                target_node.left.prev = target_node.prev
            elif target_node.left is None and target_node.right is not None:
                if is_target_left:
                    target_node.prev.left = target_node.right
                else:
                    target_node.prev.right = target_node.right
                target_node.right.prev = target_node.prev
            # remove non-root - two child case #################################
            else:
                temp_val = getImP(target_node.left).value
                self.remove(temp_val)
                target_node.value = temp_val


    def __str__(self):
        """String representation: a hierarchical view of the BST.

        Example:  (3)
                  / \     '[3]          The nodes of the BST are printed
                (2) (5)    [2, 5]       by depth levels. Edges and empty
                /   / \    [1, 4, 6]'   nodes are not printed.
              (1) (4) (6)
        """
        if self.root is None:                       # Empty tree
            return "[]"
        out, current_level = [], [self.root]        # Nonempty tree
        while current_level:
            next_level, values = [], []
            for node in current_level:
                values.append(node.value)
                for child in [node.left, node.right]:
                    if child is not None:
                        next_level.append(child)
            out.append(values)
            current_level = next_level
        return "\n".join([str(x) for x in out])

    def draw(self):
        """Use NetworkX and Matplotlib to visualize the tree."""
        if self.root is None:
            return

        # Build the directed graph.
        G = nx.DiGraph()
        G.add_node(self.root.value)
        nodes = [self.root]
        while nodes:
            current = nodes.pop(0)
            for child in [current.left, current.right]:
                if child is not None:
                    G.add_edge(current.value, child.value)
                    nodes.append(child)

        # Plot the graph. This requires graphviz_layout (pygraphviz).
        nx.draw(G, pos=graphviz_layout(G, prog="dot"), arrows=True,
                with_labels=True, node_color="C1", font_size=8)
        plt.show()


class AVL(BST):
    """Adelson-Velsky Landis binary search tree data structure class.
    Rebalances after insertion when needed.
    """
    def insert(self, data):
        """Insert a node containing the data into the tree, then rebalance."""
        BST.insert(self, data)      # Insert the data like usual.
        n = self.find(data)
        while n:                    # Rebalance from the bottom up.
            n = self._rebalance(n).prev

    def remove(*args, **kwargs):
        """Disable remove() to keep the tree in balance."""
        raise NotImplementedError("remove() is disabled for this class")

    def _rebalance(self,n):
        """Rebalance the subtree starting at the specified node."""
        balance = AVL._balance_factor(n)
        if balance == -2:                                   # Left heavy
            if AVL._height(n.left.left) > AVL._height(n.left.right):
                n = self._rotate_left_left(n)                   # Left Left
            else:
                n = self._rotate_left_right(n)                  # Left Right
        elif balance == 2:                                  # Right heavy
            if AVL._height(n.right.right) > AVL._height(n.right.left):
                n = self._rotate_right_right(n)                 # Right Right
            else:
                n = self._rotate_right_left(n)                  # Right Left
        return n

    @staticmethod
    def _height(current):
        """Calculate the height of a given node by descending recursively until
        there are no further child nodes. Return the number of children in the
        longest chain down.
                                    node | height
        Example:  (c)                  a | 0
                  / \                  b | 1
                (b) (f)                c | 3
                /   / \                d | 1
              (a) (d) (g)              e | 0
                    \                  f | 2
                    (e)                g | 0
        """
        if current is None:     # Base case: the end of a branch.
            return -1           # Otherwise, descend down both branches.
        return 1 + max(AVL._height(current.right), AVL._height(current.left))

    @staticmethod
    def _balance_factor(n):
        return AVL._height(n.right) - AVL._height(n.left)

    def _rotate_left_left(self, n):
        temp = n.left
        n.left = temp.right
        if temp.right:
            temp.right.prev = n
        temp.right = n
        temp.prev = n.prev
        n.prev = temp
        if temp.prev:
            if temp.prev.value > temp.value:
                temp.prev.left = temp
            else:
                temp.prev.right = temp
        if n is self.root:
            self.root = temp
        return temp

    def _rotate_right_right(self, n):
        temp = n.right
        n.right = temp.left
        if temp.left:
            temp.left.prev = n
        temp.left = n
        temp.prev = n.prev
        n.prev = temp
        if temp.prev:
            if temp.prev.value > temp.value:
                temp.prev.left = temp
            else:
                temp.prev.right = temp
        if n is self.root:
            self.root = temp
        return temp

    def _rotate_left_right(self, n):
        temp1 = n.left
        temp2 = temp1.right
        temp1.right = temp2.left
        if temp2.left:
            temp2.left.prev = temp1
        temp2.prev = n
        temp2.left = temp1
        temp1.prev = temp2
        n.left = temp2
        return self._rotate_left_left(n)

    def _rotate_right_left(self, n):
        temp1 = n.right
        temp2 = temp1.left
        temp1.left = temp2.right
        if temp2.right:
            temp2.right.prev = temp1
        temp2.prev = n
        temp2.right = temp1
        temp1.prev = temp2
        n.right = temp2
        return self._rotate_right_right(n)


# Problem 4
def prob4():
    """Compare the build and search times of the SinglyLinkedList, BST, and
    AVL classes. For search times, use SinglyLinkedList.iterative_find(),
    BST.find(), and AVL.find() to search for 5 random elements in each
    structure. Plot the number of elements in the structure versus the build
    and search times. Use log scales where appropriate.
    """
    # read english.txt and populate word list ##################################
    word_list = []
    with open("english.txt", 'r') as my_infile:
        linelist = my_infile.readlines()
        for i in range(len(linelist)):
            word_list.append(str(linelist[i]).rstrip())
    # initialize domain and times lists ########################################
    domain = [2**i for i in range(3,11)]
    build_linked_list = []
    build_bst = []
    build_avl = []
    find_linked_list = []
    find_bst = []
    find_avl = []

    # use for-loop to time construction and traversal ##########################
    for n in domain:
        # construct necessary item lists #######################################
        #random_list = []
        #for i in range(n):  # create list of length n of random items
        #    random_list.append(rand.choice(word_list))
        random_list = rand.sample(set(word_list), n)
        #search_list = []
        #for i in range(5):
        #    search_list.append(rand.choice(random_list))
        search_list = rand.sample(set(random_list), 5)

        # initialize data structures ###########################################
        my_list = SinglyLinkedList()
        my_bst = BST()
        my_avl = AVL()

        # time linked list construction ########################################
        start_time = time.time()
        for item in random_list:
            my_list.append(item)
        build_linked_list.append(time.time() - start_time)
        # time bst construction ################################################
        start_time = time.time()
        for item in random_list:
            my_bst.insert(item)
        build_bst.append(time.time() - start_time)
        # time avl construction ################################################
        start_time = time.time()
        for item in random_list:
            my_avl.insert(item)
        build_avl.append(time.time() - start_time)

        # time linked list find ################################################
        start_time = time.time()
        for item in search_list:
            my_list.iterative_find(item)
        find_linked_list.append(time.time() - start_time)
        # time bst find ########################################################
        start_time = time.time()
        for item in search_list:
            my_bst.find(item)
        find_bst.append(time.time() - start_time)
        # time avl find ########################################################
        start_time = time.time()
        for item in search_list:
            my_avl.find(item)
        find_avl.append(time.time() - start_time)

    # plot and print results ###################################################
    ax1 = plt.subplot(121)
    ax1.loglog(domain, build_linked_list, '.-', color='orange', basex=2, basey=2,linewidth=2, markersize=5, label='Linked List')
    ax1.loglog(domain, build_bst, '.-', color='blue', basex=2, basey=2,linewidth=2, markersize=5, label='BST')
    ax1.loglog(domain, build_avl, '.-', color='green', basex=2, basey=2,linewidth=2, markersize=5, label='AVL')
    ax1.legend(loc="upper left")
    plt.xlabel("Size", fontsize=14)
    plt.ylabel("Time", fontsize=14)
    plt.title("Build Times")
    ax2 = plt.subplot(122)
    ax2.loglog(domain, find_linked_list, '.-', color='orange', basex=2, basey=2,linewidth=2, markersize=5, label='Linked List')
    ax2.loglog(domain, find_bst, '.-', color='blue', basex=2, basey=2,linewidth=2, markersize=5, label='BST')
    ax2.loglog(domain, find_avl, '.-', color='green', basex=2, basey=2,linewidth=2, markersize=5, label='AVL')
    ax2.legend(loc="upper left")
    plt.xlabel("Size", fontsize=14)
    plt.ylabel("Time", fontsize=14)
    plt.title("Find Times")
    plt.show()