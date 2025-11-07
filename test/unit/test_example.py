# -*- coding: utf-8 -*-
"""
module: Unit tests
"""


def calculate(one, two):
    """Return the sum of two numbers"""
    return one + two


def test_calculate():
    """Return a test of a sum function"""
    assert calculate(3, 4) == 7
