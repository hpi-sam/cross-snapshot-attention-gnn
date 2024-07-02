# Dana Angluin's algorithm for generating expander graphs

# Author:   Vlad Burca
# Date:     November 23, 2013
# Updated:  November 24, 2013


# ANGLUIN EXPANDERS
# Description:
# Generates the H matrix from matrices A and B.
# H is an adjacency matrix constructed by adding edges between nodes in A and B
# on the following rules:
#   (x, y) of A is connected to the following nodes of B:
#     (x, y)
#     (x + y, y)
#     (y + 1, -x)


import numpy

from src.generation.behavior.expanders import helpers
from src.generation.behavior.expanders.methods import ANGLUIN


NAME = "[" + ANGLUIN.upper() + "]"
K = 3


def generate_expander(size, A_indices, n):
    size_H = 2 * size
    H = numpy.empty(
        shape=(size_H, K), dtype=numpy.int32
    )  # Generate H, empty adjacency list matrix

    for row in A_indices:
        for element_index in row:  # Get the tuple index from the matrix of indices (A)
            x0 = element_index // n  # Grab first value
            y0 = element_index % n  # Grab second value

            i = element_index  # Grab the index of the (x0, y0) element

            # connect to (x, y) in B
            x = x0
            y = y0
            j = (x * n + y % n) + size  # add the shift in the H indexing

            H[i][0] = j  # node with index i is connected to node with index j
            H[j][0] = i  # vice-versa

            # connect to (x + y, y) in B
            x = (x0 + y0) % n
            y = y0
            j = (x * n + y % n) + size

            H[i][1] = j
            H[j][1] = i

            # connect to (y + 1, -x) in B
            x = (y0 + 1) % n
            y = (-x0) % n
            j = (x * n + y % n) + size

            H[i][2] = j
            H[j][2] = i
    return H


def GENERATE_ANGLUIN_EXPANDERS(size, A_indices, n, EPSILON):
    size_H = 2 * size
    H = generate_expander(size, A_indices, n)

    print(
        NAME
        + " Generating H (adjacency list matrix) of size "
        + str(size_H)
        + " x "
        + str(K)
        + " ... "
    )

    print(NAME + " Generated adjacency list matrix H.")

    print(NAME + " Calculating second highest eigenvalue of H ... ")

    eigenvalue = helpers.generate_eigenvalue(H, size_H, K, EPSILON, NAME)

    print(NAME + " Calculated second highest eigenvalue of H.")

    helpers.write_result(NAME, size_H, K, eigenvalue)
    helpers.cleanup(".aux")
