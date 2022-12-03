# shell2.py
"""Volume 3: Unix Shell 2.
Marcelo Leszynski
Math 345 sec 005
11/10/20
"""
import os
import subprocess
from glob import glob

# Problem 3
def grep(target_string, file_pattern):
    """Find all files in the current directory or its subdirectories that
    match the file pattern, then determine which ones contain the target
    string.

    Parameters:
        target_string (str): A string to search for in the files whose names
            match the file_pattern.
        file_pattern (str): Specifies which files to search.
    """
    files = glob("**/"+file_pattern, recursive=True)  # create a list of desired file types   
    tru_files = []  
    for fil in files:  # for files of the desired filetype
        string = subprocess.check_output(['cat',  fil]).decode()  # create string representation of file contents
        if string.find(target_string) != -1:  # check if file string has target_string
            tru_files.append(fil)
    return tru_files


# Problem 4
def largest_files(n):
    """Return a list of the n largest files in the current directory or its
    subdirectories (from largest to smallest).
    """
    g = subprocess.check_output("""find "$(pwd)" -type f -ls | sort -rh -k 7 | awk '{print $11}'| head -""" + str(n), shell= True).decode()
    split = g.split("\n")
    split.pop()
    subprocess.call("wc -l < " + str(split[-1]) + " > ./smallest.txt", shell=True)
   
    return split
    
# Problem 6    
def prob6(n = 10):
   """this problem counts to or from n three different ways, and
      returns the resulting lists each integer
   
   Parameters:
       n (int): the integer to count to and down from
   Returns:
       integerCounter (list): list of integers from 0 to the number n
       twoCounter (list): list of integers created by counting down from n by two
       threeCounter (list): list of integers created by counting up to n by 3
   """
   #print what the program is doing
   integerCounter = list()
   twoCounter = list()
   threeCounter = list()
   counter = n
   for i in range(n+1):
       integerCounter.append(i)
       if (i % 2 == 0):
           twoCounter.append(counter - i)
       if (i % 3 == 0):
           threeCounter.append(i)
   #return relevant values
   return integerCounter, twoCounter, threeCounter