# linked_lists.py
"""Volume 2: Linked Lists.
Marcelo Leszynski
Math 321 Sec 005
10/01/20
"""


# Problem 1
class Node:
    """A basic node class for storing data."""
    def __init__(self, data):
        """Store the data in the value attribute. Data must be 
        of type int, float, or str.
                
        Raises:
            TypeError: if data is not of type int, float, or str.
        """
        if type(data) != (int) and type(data) != (float) and type(data) != (str):
            raise TypeError("Data must be of type int, float, or str.\n")
        self.value = data


class LinkedListNode(Node):
    """A node class for doubly linked lists. Inherits from the Node class.
    Contains references to the next and previous nodes in the linked list.
    """
    def __init__(self, data):
        """Store the data in the value attribute and initialize
        attributes for the next and previous nodes in the list.
        """
        Node.__init__(self, data)       # Use inheritance to set self.value.
        self.next = None                # Reference to the next node.
        self.prev = None                # Reference to the previous node.


# Problems 2-5
class LinkedList:
    """Doubly linked list data structure class.

    Attributes:
        head (LinkedListNode): the first node in the list.
        tail (LinkedListNode): the last node in the list.
        size (int): the current number of nodes in the linked list.
    """
    def __init__(self):
        """Initialize the head and tail attributes by setting
        them to None, since the list is empty initially.
        """
        self.head = None
        self.tail = None
        self.size = 0

    def append(self, data):
        """Append a new node containing the data to the end of the list."""
        # Create a new node to store the input data.
        new_node = LinkedListNode(data)
        if self.head is None:
            # If the list is empty, assign the head and tail attributes to
            # new_node, since it becomes the first and last node in the list.
            self.head = new_node
            self.tail = new_node
        else:
            # If the list is not empty, place new_node after the tail.
            self.tail.next = new_node               # tail --> new_node
            new_node.prev = self.tail               # tail <-- new_node
            # Now the last node in the list is new_node, so reassign the tail.
            self.tail = new_node
        self.size += 1

    # Problem 2
    def find(self, data):
        """Return the first node in the list containing the data.

        Raises:
            ValueError: if the list does not contain the data.

        Examples:
            >>> l = LinkedList()
            >>> for x in ['a', 'b', 'c', 'd', 'e']:
            ...     l.append(x)
            ...
            >>> node = l.find('b')
            >>> node.value
            'b'
            >>> l.find('f')
            ValueError: <message>
        """
        if self.head is None:
            raise ValueError("The linked list is empty. Cannot perform search.\n")
        node_iter = self.head
        while node_iter is not None:
            if node_iter.value == data:
                return node_iter
            else:
                node_iter = node_iter.next
        raise ValueError("Data not found in linked list.\n")

        


    # Problem 2
    def get(self, i):
        """Return the i-th node in the list.

        Raises:
            IndexError: if i is negative or greater than or equal to the
                current number of nodes.

        Examples:
            >>> l = LinkedList()
            >>> for x in ['a', 'b', 'c', 'd', 'e']:
            ...     l.append(x)
            ...
            >>> node = l.get(3)
            >>> node.value
            'd'
            >>> l.get(5)
            IndexError: <message>
        """
        if i < 0 or i >= self.size:  # check for valid index
            raise IndexError("Incorrect index given for get().\n")

        node_iter = self.head  # create an iterator for our linked list
        for temp in range (i):
            node_iter = node_iter.next  # step through our list until we are at desired index

        return node_iter  # return the node at the desired index

    # Problem 3
    def __len__(self):
        """Return the number of nodes in the list.

        Examples:
            >>> l = LinkedList()
            >>> for i in (1, 3, 5):
            ...     l.append(i)
            ...
            >>> len(l)
            3
            >>> l.append(7)
            >>> len(l)
            4
        """
        return self.size  # trivial. Handled by adding a size attribute to our class.

    # Problem 3
    def __str__(self):
        """String representation: the same as a standard Python list.

        Examples:
            >>> l1 = LinkedList()       |   >>> l2 = LinkedList()
            >>> for i in [1,3,5]:       |   >>> for i in ['a','b',"c"]:
            ...     l1.append(i)        |   ...     l2.append(i)
            ...                         |   ...
            >>> print(l1)               |   >>> print(l2)
            [1, 3, 5]                   |   ['a', 'b', 'c']
        """
        if self.size == 0:  # handle the empty list case
            return "[]"
        output = "["
        for i in range(self.size - 1):  # add all but last element to output
            output += repr(self.get(i).value)
            output += ', '
        output += (repr(self.get(self.size - 1).value) + ']')  # add last element to output
        return output

    # Problem 4
    def remove(self, data):
        """Remove the first node in the list containing the data.

        Raises:
            ValueError: if the list is empty or does not contain the data.

        Examples:
            >>> print(l1)               |   >>> print(l2)
            ['a', 'e', 'i', 'o', 'u']   |   [2, 4, 6, 8]
            >>> l1.remove('i')          |   >>> l2.remove(10)
            >>> l1.remove('a')          |   ValueError: <message>
            >>> l1.remove('u')          |   >>> l3 = LinkedList()
            >>> print(l1)               |   >>> l3.remove(10)
            ['e', 'o']                  |   ValueError: <message>
        """
        target_node = self.find(data)
        if target_node is self.head and target_node is self.tail:  # target node is the only node in list
            self.head = None
            self.tail = None
        elif target_node is self.head:  # target node is the head of the list
            self.head = self.head.next
            self.head.prev = None
        elif target_node is self.tail:  # target node is the tail of the list
            self.tail = self.tail.prev
            self.tail.next = None
        else:  # target node is not the first, last, or only element in the list
            target_node.prev.next = target_node.next
            target_node.next.prev = target_node.prev
            target_node = None
        self.size -= 1  # decrement size of list to account for removal

    # Problem 5
    def insert(self, index, data):
        """Insert a node containing data into the list immediately before the
        node at the index-th location.

        Raises:
            IndexError: if index is negative or strictly greater than the
                current number of nodes.

        Examples:
            >>> print(l1)               |   >>> len(l2)
            ['b']                       |   5
            >>> l1.insert(0, 'a')       |   >>> l2.insert(6, 'z')
            >>> print(l1)               |   IndexError: <message>
            ['a', 'b']                  |
            >>> l1.insert(2, 'd')       |   >>> l3 = LinkedList()
            >>> print(l1)               |   >>> l3.insert(1, 'a')
            ['a', 'b', 'd']             |   IndexError: <message>
            >>> l1.insert(2, 'c')       |
            >>> print(l1)               |
            ['a', 'b', 'c', 'd']        |
        """
        if index < 0 or index > self.size:
            raise IndexError("Index value is invalid.\n")
        if index == self.size:  # case of adding a last element
            self.append(data)
        elif index == 0:  # case of adding a first element
            new_node = LinkedListNode(data)  # initialize a new node
            new_node.next = self.head  # set the next of the new node to the old head
            self.head.prev = new_node  # set the prev of the old head to the new node
            self.head = new_node  # set head to the new node
        else:  # case of adding an element in the middle of the list
            new_node = LinkedListNode(data)  # initialize a new node
            swap_node = self.get(index)  # create a pointer to the node at the given index
            new_node.next = swap_node 
            new_node.prev = swap_node.prev
            new_node.prev.next = new_node
            new_node.next.prev = new_node
        self.size += 1


# Problem 6: Deque class.
class Deque(LinkedList):
    def __init__(self):
        LinkedList.__init__(self)  # use the inherited LinkedList constructor

    def pop(self):
        if self.size <= 0:  # handle the case where the deque is empty
            raise ValueError("The deque is empty.\n")
        elif self.size == 1:  # handle the case where the deque is of size 1
            target_node = self.head
            self.head = None  # set deque head and tail to None
            self.tail = None
            self.size = 0  # deque is now size zero
            return target_node.value
        else:
            target_node = self.tail  # create temporary pointer
            self.tail = target_node.prev  # move tail to tail.prev
            self.tail.next = None  # make tail the end of the deque
            self.size -= 1
            return target_node.value

    def popleft(self):
        if self.size <= 0:  # handle the case where the deque is empty
            raise ValueError("The deque is empty.\n")
        elif self.size == 1:  # handle the case where the deque is of size 1
            target_node = self.head
            self.head = None  # set deque head and tail to None
            self.tail = None
            self.size = 0  # deque is now size zero
            return target_node.value
        else:
            target_node = self.head  # create a temporary pointer
            self.head = target_node.next  # move head to head.next
            self.head.prev = None  # make head the beginning of the deque
            self.size -= 1
            return target_node.value

    def appendleft(self, data):
        """Append a new node containing the data to the end of the list."""
        # Create a new node to store the input data.
        new_node = LinkedListNode(data)
        if self.head is None:
            # If the list is empty, assign the head and tail attributes to
            # new_node, since it becomes the first and last node in the list.
            self.head = new_node
            self.tail = new_node
        else:
            # If the list is not empty, place new_node after the tail.
            new_node.next = self.head
            self.head.prev = new_node
            self.head = new_node
        self.size += 1

    def remove(self, *args, **kwargs):
        raise NotImplementedError("Use pop() or popleft() for removal")

    def insert(self, *args, **kwargs):
        raise NotImplementedError("Use append() or appendleft() to insert nodes")

       



# Problem 7
def prob7(infile, outfile):
    """Reverse the contents of a file by line and write the results to
    another file.

    Parameters:
        infile (str): the file to read from.
        outfile (str): the file to write to.
    """
    my_stack = Deque()  # use a deque as a stack to push and pop text lines
    with open(infile, 'r') as my_infile:
        linelist = my_infile.readlines()  # read in text from file
        for i in range(len(linelist)):  # store text in object
            my_stack.append(str(linelist[i]))
        my_stack.tail.value += "\n"
        my_stack.head.value = my_stack.head.value.rstrip()
    with open(outfile, 'w') as my_outfile:
        for i in range(my_stack.size):
            my_outfile.write(my_stack.pop())  # pop stored text to outfile in reverse order