# test_specs.py
"""Python Essentials: Unit Testing.
Marcelo Leszynski
Math 321 Sec 005
09/24/20
"""

import specs
import pytest


def test_add():
    assert specs.add(1, 3) == 4, "failed on positive integers"
    assert specs.add(-5, -7) == -12, "failed on negative integers"
    assert specs.add(-6, 14) == 8

def test_divide():
    assert specs.divide(4,2) == 2, "integer division"
    assert specs.divide(5,4) == 1.25, "float division"
    with pytest.raises(ZeroDivisionError) as excinfo:
        specs.divide(4, 0)
    assert excinfo.value.args[0] == "second input cannot be zero"


# Problem 1: write a unit test for specs.smallest_factor(), then correct it.
def test_smallest_factor():
    assert specs.smallest_factor(1) == 1, "failed on input 1"
    assert specs.smallest_factor(2) == 2, "failed on input 2"
    assert specs.smallest_factor(3) == 3, "failed on input 3"
    assert specs.smallest_factor(4) == 2, "failed on input 4"
    assert specs.smallest_factor(15) == 3, "failed on input 15"
    assert specs.smallest_factor(75) == 3, "failed on input 75"


# Problem 2: write a unit test for specs.month_length().
def test_month_length():
    assert specs.month_length("September") == 30, "failed on September non-leap-year"
    assert specs.month_length("April") == 30, "failed on April non-leap-year"
    assert specs.month_length("June") == 30, "failed on June non-leap-year"
    assert specs.month_length("November") == 30, "failed on November non-leap-year"
    assert specs.month_length("September", True) == 30, "failed on September leap-year"
    assert specs.month_length("April", True) == 30, "failed on April leap-year"
    assert specs.month_length("June", True) == 30, "failed on June leap-year"
    assert specs.month_length("November", True) == 30, "failed on November leap-year"
    assert specs.month_length("January") == 31, "failed on January non-leap-year"
    assert specs.month_length("January", True) == 31, "failed on January leap-year"
    assert specs.month_length("March") == 31, "failed on March non-leap-year"
    assert specs.month_length("March", True) == 31, "failed on March leap-year"
    assert specs.month_length("May") == 31, "failed on May non-leap-year"
    assert specs.month_length("May", True) == 31, "failed on May leap-year"
    assert specs.month_length("July") == 31, "failed on July non-leap-year"
    assert specs.month_length("July", True) == 31, "failed on July leap-year"
    assert specs.month_length("August") == 31, "failed on August non-leap-year"
    assert specs.month_length("August", True) == 31, "failed on August leap-year"
    assert specs.month_length("October") == 31, "failed on October non-leap-year"
    assert specs.month_length("October", True) == 31, "failed on October leap-year"
    assert specs.month_length("December") == 31, "failed on December non-leap-year"
    assert specs.month_length("December", True) == 31, "failed on December leap-year"
    assert specs.month_length("February") == 28, "failed on February non-leap-year"
    assert specs.month_length("February", True) == 29, "failed on February leap-year"
    assert specs.month_length("squeeps") == None, "failed on incorrect input"


# Problem 3: write a unit test for specs.operate().
def test_operate():
    with pytest.raises(TypeError) as excinfo:
        specs.operate(4, 0, 12)
    assert excinfo.value.args[0] == "oper must be a string", "failed on non-string operator"
    assert specs.operate(4, 2, '+') == 6, "failed on + operator"
    assert specs.operate(4, 2, '-') == 2, "failed on - operator"
    assert specs.operate(4, 2, '*') == 8, "failed on * operator"
    assert specs.operate(4, 2, '/') == 2, "failed on / operator"
    with pytest.raises(ZeroDivisionError) as divinfo:
        specs.operate(4, 0, '/')
    assert divinfo.value.args[0] == "division by zero is undefined", "failed on division by zero"
    with pytest.raises(ValueError) as valinfo:
        specs.operate(4, 2, "12")
    assert valinfo.value.args[0] == "oper must be one of '+', '/', '-', or '*'", "failed on incorrect operator"

# Problem 4: write unit tests for specs.Fraction, then correct it.
@pytest.fixture
def set_up_fractions():
    frac_1_3 = specs.Fraction(1, 3)
    frac_1_2 = specs.Fraction(1, 2)
    frac_n2_3 = specs.Fraction(-2, 3)
    return frac_1_3, frac_1_2, frac_n2_3

def test_fraction_init(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert frac_1_3.numer == 1
    assert frac_1_2.denom == 2
    assert frac_n2_3.numer == -2
    frac = specs.Fraction(30, 42)
    assert frac.numer == 5
    assert frac.denom == 7
    with pytest.raises(ZeroDivisionError) as divinfo:
        specs.Fraction(1, 0)
    assert divinfo.value.args[0] == "denominator cannot be zero"
    with pytest.raises(TypeError) as numinfo:
        specs.Fraction('n', 2)
    assert numinfo.value.args[0] == "numerator and denominator must be integers"
    with pytest.raises(TypeError) as denominfo:
        specs.Fraction(1, '2')
    assert denominfo.value.args[0] == "numerator and denominator must be integers"
    with pytest.raises(TypeError) as bothinfo:
        specs.Fraction('l', 'm')
    assert bothinfo.value.args[0] == "numerator and denominator must be integers"


def test_fraction_str(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert str(frac_1_3) == "1/3"
    assert str(frac_1_2) == "1/2"
    assert str(frac_n2_3) == "-2/3"
    assert str(specs.Fraction(3, 1)) == "3"
    assert str(specs.Fraction(0, 1)) == "0"


def test_fraction_float(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert float(frac_1_3) == 1 / 3.
    assert float(frac_1_2) == .5
    assert float(frac_n2_3) == -2 / 3.
    assert float(specs.Fraction(16, 2)) == 8.


def test_fraction_eq(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert frac_1_2 == specs.Fraction(1, 2)
    assert frac_1_3 == specs.Fraction(2, 6)
    assert frac_n2_3 == specs.Fraction(8, -12)
    assert specs.Fraction(1, 2) == 0.5
    assert specs.Fraction(1, 2) != 1.2


def test_fraction_add(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert frac_1_2 + frac_1_3 == specs.Fraction(5, 6)
    assert frac_n2_3 + frac_1_3 == specs.Fraction(-1, 3)

def test_fraction_sub(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert frac_1_2 - frac_1_3 == specs.Fraction(1, 6)
    assert frac_n2_3 - frac_n2_3 == specs.Fraction(0, 1)

def test_fraction_mul(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert frac_1_2 * frac_1_3 == specs.Fraction(1, 6)
    assert frac_n2_3 * frac_n2_3 == specs.Fraction(4, 9)

def test_fraction_truediv(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert frac_1_2 / frac_1_3 == specs.Fraction(3, 2)
    assert frac_n2_3 / frac_n2_3 == specs.Fraction(1, 1)
    with pytest.raises(ZeroDivisionError) as zeroinf:
        specs.Fraction(1, 2) / specs.Fraction(0, 3)
    assert zeroinf.value.args[0] == "cannot divide by zero"


# Problem 5: Write test cases for Set.
@pytest.fixture
def setup_bad_sets():
    set_good = ['0120', '1200', '2001', '0012', '0102', '1020', '0201', '2010', '0210', '2100', '1002', '0021']
    set_11_cards = ['0120', '1200', '2001', '0012', '0102', '1020', '0201', '2010', '0210', '2100', '1002']
    set_card_multiples = ['0120', '1200', '2001', '0012', '0120', '1020', '0201', '2010', '0210', '2100', '1002', '0021']
    set_card_3digits = ['0120', '1200', '2001', '0012', '0102', '1020', '0201', '010', '0210', '2100', '1002', '0021']
    set_bad_char = ['0120', '1200', '2001', '0012', '0132', '1020', '0201', '2010', '0210', '2100', '1002', '0021']
    return set_good, set_11_cards, set_card_multiples, set_card_3digits, set_bad_char

def test_set_count_sets(setup_bad_sets):
    set_good, set_11_cards, set_card_multiples, set_card_3digits, set_bad_char = setup_bad_sets
    with pytest.raises(ValueError) as lesscards:
        specs.count_sets(set_11_cards)
    assert lesscards.value.args[0] == "List must contain exactly 12 cards"
    with pytest.raises(ValueError) as multiples:
        specs.count_sets(set_card_multiples)
    assert multiples.value.args[0] == "List cannot contain repeat cards"
    with pytest.raises(ValueError) as not4digits:
        specs.count_sets(set_card_3digits)
    assert not4digits.value.args[0] == "Cards must have exactly four digits"
    with pytest.raises(ValueError) as not4digits2:
        specs.count_sets(['0120', '1200', '2001', '0012', '0102', '1020', '0201', '2010', '0210', '2100', '1002', '00210'])
    assert not4digits2.value.args[0] == "Cards must have exactly four digits"
    with pytest.raises(ValueError) as wrongchar:
        specs.count_sets(set_bad_char)
    assert wrongchar.value.args[0] == "Cards can only contain '0', '1', or '2'"
    assert specs.count_sets(set_good) == 12

def test_set_is_set():
    assert specs.is_set('0123', '1230', '2310') == True, "failed on is_set('0123', '1230', '2301')"
    assert specs.is_set('0123', '0123', '0123') == False, "failed on is_set('0123', '0123', '0123')"
    assert specs.is_set('0123', '1230', '1301') == False, "failed on is_set('0123', '1230', '1301')"
