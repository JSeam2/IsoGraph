"""
Implementation referenced from
https://github.com/handcraftsman/GeneticAlgorithmsWithPython/blob/master/ch02/genetic.py
"""

import random
from qutip import *
import numpy as np
import pandas as pd
from functools import reduce
import datetime
import time


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


def generate_initial_population(N, data, population_size = 20, depth = 5):
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

    fitness, acc = get_fitness(genes, data)

    return PopulationPool(genes, fitness, acc)


def generate_children(parent, data, take_best,
                      population_size = 20,
                      mutation_rate = 0.05):
    """
    produce children with mutations

    parent: PopulationPool class
    mutation_rate: float probability that value will be mutated
    crossover_rate: float probability that genes will cross with another gene
    """
    child_genes = []

    best_parent_genes = parent.genes[:take_best]
    while len(child_genes) < population_size:
        # randomly pick a parent
        parentA_gene = random.choice(best_parent_genes)

        # randomly pick another parent
        parentB_gene = random.choice(best_parent_genes)

        # crossover the gene at a random point
        rand_point = random.randint(0, parentA_gene.shape[0])

        if parentB_gene.shape[0] < rand_point:
            child_gene = np.vstack((parentA_gene[0:rand_point],
                                    parentB_gene[rand_point, parentB_gene.shape[0]]))

        else :
            child_gene = parentA_gene

        if random.random() <= mutation_rate:
            # randomly change values in the array
            mask = np.random.randint(0,2,size=child_gene.shape).astype(np.bool)
            r = np.random.uniform(-np.pi, np.pi, size = child_gene.shape)
            child_gene[mask] = r[mask]

        child_genes.append(child_gene)

    fitness, acc = get_fitness(child_genes, data)

    return PopulationPool(child_genes, fitness, acc)


def get_fitness(genes, data):
    """
    Pass in gene and run through the various isomorphic graphs

    gene: list of np array
    data: panda dataframe from pkl file

    returns list of fitness
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

    fitness_list = []
    acc_list = []

    for gene in genes:
        loss = 0
        correct = 0

        # make circuit using the genes
        circuit = make_circuit(gene)

        for index, row in data.iterrows():
            if index % 2500 == 0:
                print("running {}".format(index))

            combined = row["G1"][upper].tolist()[0] + row["G2"][upper].tolist()[0]
            int_comb = [int(i) for i in combined]
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

            if expectation <= 0.50:
                # this is 1
                prediction = 1

            else:
                prediction = 0

            if prediction == actual:
                correct += 1


        ave_loss = loss/num_sample
        fitness_list.append(ave_loss)

        accuracy = correct/num_sample
        acc_list.append(accuracy)

    return fitness_list, acc_list


def get_best(N, data, num_epoch = 10,
             population_size = 20,
             take_best = 5,
             depth = 5,
             mutation_rate = 0.05):

    """
    N refers to the number of nodes
    Population size refers to the number of individuals in the population
    Take_best refers to the number of top individuals we take
    depth refers to how deep the quantum circuit should go
    mutation_rate refers to the probability of children mutating
    """
    assert take_best >= 2
    assert population_size >= 2
    assert take_best <= population_size

    def display(pool):
        print("Time: {} \t Best Score: {} \t Best Acc: {}".format(datetime.datetime.now(),
                                                            pool.fitness[0],
                                                            pool.accuracy[0]))

    parent = generate_initial_population(N, data, population_size, depth)
    parent.sort()
    display(parent)

    # take best

    for i in range(num_epoch):
        child = generate_children(parent, data, take_best, population_size,
                                  mutation_rate)
        child.sort()
        print()
        print("Epoch {}".format(i))
        display(child)
        parent = child


class PopulationPool:
    def __init__(self, genes, fitness, accuracy):
        """
        genes : list of genes
        fitness : list of fitness
        accuracy : list of accuracy
        """
        self.genes = genes
        self.fitness = fitness
        self.accuracy = accuracy

    def sort(self):
        """
        returns list of genes sorted by fitness in increasing order

        """
        self.genes = [x for _,x in sorted(zip(self.fitness, self.genes))]
        self.accuracy = [x for _,x in sorted(zip(self.fitness, self.accuracy))]


if __name__ == "__main__":
    df = pd.read_pickle("3_node_10000.pkl")
    #initial_genes = generate_initial_population(N = 3, population_size = 2)

    #upper = np.triu_indices(3, 1)

    #comb = df["G1"][0][upper].tolist()[0] + df["G2"][0][upper].tolist()[0]
    #int_comb = [int(i) for i in comb]

    #theta = np.random.uniform(-np.pi, np.pi, [2, 3*2])

    #start = time.time()
    #print(get_fitness([theta], df))
    #print("Time taken: {}".format(time.time()- start))

    #circuit = make_circuit(theta)
    #inputval = bra(int_comb)
    #result = inputval* circuit
    #density = result.dag() * result

    ## wait a minute is this the right side?
    #projector = tensor(basis(2,0) * basis(2,0).dag(), identity(2), identity(2),
    #                   identity(2), identity(2), identity(2))

    #expectation = expect(projector, density)
    #print(expectation)

    get_best(N=3, data = df, num_epoch = 10, population_size = 10, take_best = 5, depth=10)
