"""
With this workflow script all relevant validation scripts can be run. It includes the following:
    test_GLOFAS:    comparing the GLOFAS dataset against other data products to check alignment.

Eventually we should turn this script into a parser, so that it contains the means to define how to call it from the command line (?)
"""

from ..validation.river_input.test_GLOFAS import plot_glofas_lin
from ..utils.validation.modify_delta_masks import *


def test_glofas(test_id):
    plot_glofas_lin(test_id)
