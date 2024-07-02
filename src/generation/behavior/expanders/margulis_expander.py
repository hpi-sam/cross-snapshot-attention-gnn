# G.A Margulis' algorithm for generating expander graphs

# Author:   Vlad Burca
# Date:     November 23, 2013
# Updated:  November 24, 2013


# MARGULIS EXPANDERS
# Description:
# Generates the H matrix from matrices A and B.
# H is an adjacency matrix constructed by adding edges between nodes in A and B
# on the following rules:
#   (x, y) of A is connected to the following nodes of B:
#     (x, y)
#     (x + 1, y)
#     (x, y + 1)
#     (x + y, y)
#     (-y, x)


import numpy

from src.generation.behavior.expanders import helpers
from src.generation.behavior.expanders.methods import MARGULIS


NAME = "[" + MARGULIS.upper() + "]"
K = 5


def generate_expander(size, A_indices, n):
    size_H = 2 * size
    H = numpy.empty(
        shape=(size_H, K), dtype=numpy.int32
    )  # Generate H, empty adjacency list matrix

    for row in A_indices:
        for element_index in row:  # Get the tuple index from the matrix of indices (A)
            x0 = element_index // n  # Grab first value
            y0 = element_index % n  # Grab second value

            i = element_index  # Grab the index of the (x, y) element

            # connect to (x, y) in B
            x = x0
            y = y0
            j = (x * n + y % n) + size  # add the shift in the H indexing
            H[i][0] = j  # node with index i is connected to node with index j
            H[j][0] = i  # vice-versa

            # connect to (x + 1, y) in B
            x = (x0 + 1) % n
            y = y0
            j = (x * n + y % n) + size

            H[i][1] = j
            H[j][1] = i

            # connect to (x, y + 1) in B
            x = x0
            y = (y0 + 1) % n
            j = (x * n + y % n) + size

            H[i][2] = j
            H[j][2] = i

            # connect to (x + y, y) in B
            x = (x0 + y0) % n
            y = y0
            j = (x * n + y % n) + size

            H[i][3] = j
            H[j][3] = i

            # connect to (-y, x) in B
            x = (-y0) % n
            y = x0
            j = (x * n + y % n) + size

            H[i][4] = j
            H[j][4] = i

    return H


def GENERATE_MARGULIS_EXPANDERS(size, A_indices, n, EPSILON):
    size_H = size

    print(
        NAME
        + " Generating H (adjacency list matrix) of size "
        + str(size_H)
        + " x "
        + str(K)
        + " ... "
    )

    H = generate_expander(size, A_indices, n)

    print(NAME + " Generated adjacency list matrix H.")

    print(NAME + " Calculating second highest eigenvalue of H ... ")

    eigenvalue = helpers.generate_eigenvalue(H, size_H, K, EPSILON, NAME)

    print(NAME + " Calculated second highest eigenvalue of H.")

    helpers.write_result(NAME, size_H, K, eigenvalue)
    helpers.cleanup(".aux")
