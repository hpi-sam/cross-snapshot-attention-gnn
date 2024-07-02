# HELPER METHODS for the generation of Expander Graphs

# Author:   Vlad Burca
# Date:     November 23, 2013
# Updated:  April 3, 2014

import numpy
import subprocess
from numpy import linalg

from src.generation.behavior.expanders import matrix_helper
from src.generation.behavior.expanders import methods

# Check if any of the passed algorithms is configured to run.
# @param: *args - the algorthms to test for.
# @return: whether any of the passed algorithms is configured to run.
def check_configured_run(*args):
    config_file = open("config.yaml", "r")
    config_vals = {}  # yaml.safe_load(config_file)
    config_file.close()

    for algorithm in args:
        if config_vals["algorithms"][algorithm] == True:
            return True

    return False


def cleanup(extension):
    import os

    filelist = [f for f in os.listdir(".") if f.endswith(extension)]
    for f in filelist:
        os.remove(f)


def generate_pair_matrices(cross_Z, A_indices, B_indices, n):
    A_elements = list()
    for row in A_indices:
        for element in row:
            A_element = (
                element / n,
                element % n,
            )  # Re-generating the actual element pairs from the index of the cross product
            A_elements.append(A_element)
    A_elements_arr = numpy.array(
        A_elements, dtype=[("first", "<i4"), ("second", "<i4")]
    )  # Transform list into a NumPy array for further transformations
    # global A
    A = A_elements_arr.reshape((n, n))  # Reshape into two dimensional NumPy array

    B_elements = list()
    for row in B_indices:
        for element in row:
            B_element = (element / n, element % n)
            B_elements.append(B_element)
    B_elements_arr = numpy.array(
        B_elements, dtype=[("first", "<i4"), ("second", "<i4")]
    )
    # global B
    B = B_elements_arr.reshape((n, n))
    return [A, B]


def write_indices_matrices(A_indices, B_indices):
    outfile_A = open("indices_matrix_A.out", "w")
    outfile_B = open("indices_matrix_B.out", "w")

    for row in A_indices:
        for element in row:
            outfile_A.write(str(element) + " ")
        outfile_A.write("\n")
    outfile_A.close()

    for row in B_indices:
        for element in row:
            outfile_B.write(str(element) + " ")
        outfile_B.write("\n")
    outfile_B.close()


def write_pair_matrices(A, B):
    outfile_A = open("matrix_A.out", "w")
    outfile_B = open("matrix_B.out", "w")

    for row in A:
        for element in row:
            outfile_A.write(str(element) + " ")
        outfile_A.write("\n")
    outfile_A.close()

    for row in B:
        for element in row:
            outfile_B.write(str(element) + " ")
        outfile_B.write("\n")
    outfile_B.close()


def write_H_params(size_H, k, EPSILON, NAME):
    outfile_params = open("params.aux", "w")

    outfile_params.write(
        str(NAME) + " " + str(size_H) + " " + str(k) + " " + str(EPSILON)
    )
    outfile_params.close()


def write_H_matrix(H, NAME):
    outfile_H = open("matrix_H.aux", "w")

    for row in H:
        for element in row:
            outfile_H.write(str(element) + " ")
        outfile_H.write("\n")
    outfile_H.close()


def run_c_commands():
    import os, time

    # Compile the power method code only when there is no executable or the
    # source file (.c) was modified more recently than the compiled one
    if (not os.path.isfile("powermethod")) or (
        time.ctime(os.path.getmtime("powermethod.c"))
        > time.ctime(os.path.getmtime("powermethod"))
    ):
        # Compile C power method code
        p = subprocess.Popen("gcc -o powermethod powermethod.c", shell=True)
        p.communicate()

    p = subprocess.Popen("./powermethod", shell=True)
    p.communicate()


def read_from_c_results():
    result_file = open("eigenvalue.aux", "r")
    eigenvalue = result_file.read()

    result_file.close()

    return float(eigenvalue)


def generate_eigenvalue(H, size_H, k, EPSILON, NAME):
    write_H_params(size_H, k, EPSILON, NAME)
    write_H_matrix(H, NAME)

    run_c_commands()

    return read_from_c_results()


# def generate_eigenvalue_old(M, n, degree):
#   EPSILON = 0.001
#   return matrix_helper.powermethod(M, n, EPSILON, degree)


# def generate_eigenvalues(H, name):
#   eigenvalues = linalg.eigvals(H)
#   eigenvalues = numpy.sort(eigenvalues)[::-1]

#   return eigenvalues


# def write_eigenvalues(name, eigenvalues):
#   outfile_eigen = open(name.strip('[]') + "_eigenvalues.out", "w")

#   outfile_eigen.write(str(eigenvalues))
#   outfile_eigen.close()


def write_result(name, n, K, eigenvalue):
    import os
    import math

    name = name.strip("[]").lower()

    # Open file with initial results (if existent)
    if os.path.exists(name + ".results"):
        result_file = open(name + ".results", "r")
        # results = yaml.safe_load(result_file)
        result_file.close()
    else:  # file does not exist / no previous results
        results = {name: {}}

    # If files are empty
    if not "dict" in str(type(results)):
        results = {name: {}}

    # Calculate the expansion constant based on the fact that, for a good
    # Expander ("Ramanujan"), we need SecondEigenvalue <= 2*sqrt(K-1)
    # We will consider good expanders, the graphs that have a positive, high
    # value for the expansion_constant (the SecondEigenvalue has to be small).

    expansion_constant = 2 * math.sqrt(K - 1) - eigenvalue
    # Uncomment to get only eigenvalues in .results file
    # expansion_constant = eigenvalue

    # Update the new results
    results[name][n] = float(expansion_constant)

    # Open file to write new yaml dictionary
    result_file = open(name + ".results", "w")
    # result_file.write(
    #     yaml.dump(results, default_flow_style=False)
    # )  # update results yaml dictionary
    result_file.close()


# Method that calculates the rank of a node from the adjacency list matrix
def get_rank(node, matrix):
    rank = 0
    for edge in range(numpy.size(matrix[node])):
        rank = rank + 1 if matrix[node][edge] != -1 else rank

    return rank
