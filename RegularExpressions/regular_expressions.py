# regular_expressions.py
"""Volume 3: Regular Expressions.
Marcelo Leszynski
Math 323 Section 3
18 February 2021
"""

import re
import numpy as np

# Problem 1
def prob1():
    """Compile and return a regular expression pattern object with the
    pattern string "python".

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """
    return re.compile("python")

# Problem 2
def prob2():
    """Compile and return a regular expression pattern object that matches
    the string "^{@}(?)[%]{.}(*)[_]{&}$".

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """
    # compiling escape characters ##############################################
    return re.compile(r"\^\{@\}\(\?\)\[%\]\{\.\}\(\*\)\[_\]\{&\}\$")

# Problem 3
def prob3():
    """Compile and return a regular expression pattern object that matches
    the following strings (and no other strings).

        Book store          Mattress store          Grocery store
        Book supplier       Mattress supplier       Grocery supplier

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """
    # use 'or' operator ########################################################
    return re.compile(r"^(Book|Mattress|Grocery) (store|supplier)$")

# Problem 4
def prob4():
    """Compile and return a regular expression pattern object that matches
    any valid Python identifier. This needs to be fixed its a parameter
    according to the PDF

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """
    # initialize templates #####################################################
    ident = "((_|[a-zA-Z])(\w)*)"
    my_str = "\'[^\']*\'"
    my_num = "((\d)*(\.)?(\d)*)"

    # compile python identifier ################################################
    return re.compile(r"^%s(\s)*(\=(\s)*(%s|%s|%s))?$" % (ident, ident, my_num, my_str))

# Problem 5
def prob5(code):
    """Use regular expressions to place colons in the appropriate spots of the
    input string, representing Python code. You may assume that every possible
    colon is missing in the input string.

    Parameters:
        code (str): a string of Python code without any colons.

    Returns:
        (str): code, but with the colons inserted in the right places.
    """
    first_pass = re.compile(r"^(\s*)(if|elif|else|for|while|try|except|finally|with|def|class)(.*)$", re.MULTILINE)
    return first_pass.sub(r"\1\2\3:",code)

# Problem 6
def prob6(filename="fake_contacts.txt"):
    """Use regular expressions to parse the data in the given file and format
    it uniformly, writing birthdays as mm/dd/yyyy and phone numbers as
    (xxx)xxx-xxxx. Construct a dictionary where the key is the name of an
    individual and the value is another dictionary containing their
    information. Each of these inner dictionaries should have the keys
    "birthday", "email", and "phone". In the case of missing data, map the key
    to None.

    Returns:
        (dict): a dictionary mapping names to a dictionary of personal info.
    """
    # initialize templates #####################################################
    end_dict = {}
    name = r"[a-zA-Z]* ([A-Z]\. )?[a-zA-Z]*"
    birth = r"(\d{1,2})/(\d{1,2})/(\d{2})?(\d{2})"
    email = r"(\w|\.)*@(\w|\.)*"
    phone = r"(.*)(\d{3})(.*)(\d{3})-(\d{4})"

    # open and read file #######################################################
    with open(filename, 'r') as myfile:
        lines = myfile.readlines()

    # iterate through file lines ###############################################
    for line in lines:

        temp_dict = {}
        temp_name = re.compile(name)
        temp_birth = re.compile(birth)
        temp_email = re.compile(email)
        temp_phone = re.compile(phone)

        if temp_birth.search(line):
            f_birth = temp_birth.search(line).group()
            month = temp_birth.sub(r"\1", f_birth)
            day = temp_birth.sub(r"\2", f_birth)
            year = temp_birth.sub(r"20\4", f_birth)

            if len(month) == 1:
                month = "0" + month
            if len(day) == 1:
                day = "0" + day

            temp_dict["birthday"] = month + "/" + day + "/" + year
        else:
            temp_dict["birthday"] = None

        if temp_email.search(line):
            temp_dict["email"] = temp_email.search(line).group()
        else:
            temp_dict["email"] = None

        if temp_phone.search(line):
            temp_dict["phone"] = temp_phone.sub(r"(\2)\4-\5", temp_phone.search(line).group())
        else:
            temp_dict["phone"] = None

        end_dict[temp_name.search(line).group()] = temp_dict

    return end_dict
