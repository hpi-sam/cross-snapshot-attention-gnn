# ALGORITHM METHODS for the generation of Expander Graphs

# Author:   Vlad Burca
# Date:     November 23, 2013
# Updated:  April 1, 2014


from src.generation.behavior.expanders.angluin_expander import (
    GENERATE_ANGLUIN_EXPANDERS,
)
from src.generation.behavior.expanders.margulis_expander import (
    GENERATE_MARGULIS_EXPANDERS,
)
from src.generation.behavior.expanders.ajtai_expander import GENERATE_AJTAI_EXPANDERS
from src.generation.behavior.expanders.random_expander import GENERATE_RANDOM_EXPANDERS

from src.generation.behavior.expanders import methods


def EXPLICIT_METHOD(method_name, size, EPSILON, A_indices=None, n=None, s=None):
    if method_name == methods.ANGLUIN:
        GENERATE_ANGLUIN_EXPANDERS(size, A_indices, n, EPSILON)
    elif method_name == methods.MARGULIS:
        GENERATE_MARGULIS_EXPANDERS(size, A_indices, n, EPSILON)
    elif method_name == methods.AJTAI:
        GENERATE_AJTAI_EXPANDERS(size_H=size, EPSILON=EPSILON, s=s)


def RANDOM_METHOD(method_name, size_H, EPSILON, samples):
    if method_name == methods.RANDOM_3:
        GENERATE_RANDOM_EXPANDERS(K=3, size_H=size_H, EPSILON=EPSILON, samples=samples)
    elif method_name == methods.RANDOM_5:
        GENERATE_RANDOM_EXPANDERS(K=5, size_H=size_H, EPSILON=EPSILON, samples=samples)
