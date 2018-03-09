"""
Implementation referenced from
https://github.com/handcraftsman/GeneticAlgorithmsWithPython/blob/master/ch02/genetic.py
"""

import random
from qutip import *
import numpy as np
import pandas as pd
from functools import reduce


# QUTIP NOTES
# hadamard = "SNOT"
# CZ = "CSIGN"
# RZ = "RZ"

def make_circuit(theta_val, save_image = False):
    """
    Input theta values to create quantum circuit
    [layer 1], [layer 2]. so on
    the length of each layer should equal length of
    input state

    The theta list will act as our genome

    theta_val: 2D numpy array

    return 2D numpy array of matrices
    """
    qc = QubitCircuit(N = len(theta_val[0]))

    for i in range(len(theta_val)):
        # ADD H gates
        qc.add_1q_gate("SNOT", start = 0, end = qc.N)

        # add RZ theta gates
        for k in range(len(theta_val[0])):
            qc.add_1q_gate("RZ", start = k, end = k + 1,
                           arg_value = theta_val[i][k],
                           arg_label = theta_val[i][k])

        for k in range(len(theta_val[0]) - 1):
            qc.add_gate("CSIGN",
                        targets = [k],
                        controls = [len(theta_val[0]) - 1])

    # produce image
    if save_image:
        qc.png

    return reduce(lambda x, y: x * y, qc.propagators())


def generate_initial_population(N, population_size = 100, depth = 10):
    """
    population size is the number of individuals in the population
    N refers to the number of nodes
    depth refers to then number of layers

    population_size: int
    N: int
    """
    genes = []
    while len(genes) < population_size:
        genes.append(np.random.uniform(-np.pi, np.pi, [depth, N*2]))

    return genes


def get_fitness(gene, data):
    """
    Pass in gene and run through the various isomorphic graphs

    gene: np array
    data: panda dataframe from pkl file
    """
    # total number of sampeles
    num_sample = data.shape[0]

    # select upper diagonal ignoring zeros in the middle
    size = data["G1"][0].shape[0]
    upper = np.triu_indices(size, 1)

    # create projector we project to 0 standard basis
    projector = basis(2,0) * basis(2,0).dag()

    for i in range((size * 2) - 1):
        projector = tensor(projector, identity(2))

    print(projector)

    # make circuit using the genes
    circuit = make_circuit(gene)

    loss = 0

    for index, row in data.iterrows():
        if index % 1000 == 0:
            print("running {}".format(index))

        combined = row["G1"][upper].tolist()[0] + row["G2"][upper].tolist()[0]
        int_comb = [int(i) for i in comb]
        inputval = bra(int_comb)
        result = inputval * circuit
        density = result.dag() * result

        # We will use the logisitc regression loss function
        # as we are dealing with a classification problem
        # compare this expectation with result 
        # expectation here refers to the likelihood of getting 0
        expectation = expect(projector, density)
        actual = row["is_iso"]

        loss += -1 * actual * np.log(1 - expectation) \
                - (1 - actual) * np.log(expectation)


    ave_loss = loss/num_sample

    return ave_loss



class Chromosomes:
    def __init__(self, genes, fitness):
        self.genes = genes
        self.fitness = fitness



if __name__ == "__main__":
    df = pd.read_pickle("3_node.pkl")
    initial_genes = generate_initial_population(N = 3, population_size = 2)

    upper = np.triu_indices(3, 1)

    comb = df["G1"][0][upper].tolist()[0] + df["G2"][0][upper].tolist()[0]
    int_comb = [int(i) for i in comb]

    theta = np.random.uniform(-np.pi, np.pi, [2, 3*2])

    print(get_fitness(theta, df))

    #circuit = make_circuit(theta)
    #inputval = bra(int_comb)
    #result = inputval* circuit
    #density = result.dag() * result

    ## wait a minute is this the right side?
    #projector = tensor(basis(2,0) * basis(2,0).dag(), identity(2), identity(2),
    #                   identity(2), identity(2), identity(2))

    #expectation = expect(projector, density)
    #print(expectation)
