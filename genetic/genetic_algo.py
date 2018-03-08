import random
from qutip import *
import numpy as np


# QUTIP NOTES
# hadamard = "SNOT"
# CZ = "CSIGN"
# RZ = "RZ"

def make_circuit(theta_val):
    """
    Input theta values to create quantum circuit
    [layer 1], [layer 2]. so on
    the length of each layer should equal length of
    input state

    theta_val: list of list

    return list of propagators
    """
    qc = QubitCircuit(N = len(theta_val[0]))

    print(len(theta_val))
    print(len(theta_val[0]))
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
    qc.png

    return qc.propagators()




def get_fitness(genes):
    """
    Pass in genes and run through the various isomorphic graphs
    """
    pass


if __name__ == "__main__":
    theta_val = [[1,1,1,1,1,1],
                 [1,1,1,1,1,1],
                 [1,1,1,1,1,1]]
    out = make_circuit(theta_val)
    print(out)
