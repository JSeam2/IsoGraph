from qutip import *
import numpy as np

a = Qobj([[1],[0]])
b = Qobj([[0],[1]])

N = 2
qc = QubitCircuit(N)
qc.add_gate(r"SNOT",r"SNOT",0)
qc.add_gate(r"RZ",0,None, np.pi/2,r'pi/2') 
qc.add_gate(r"cz",1,1)

qc.add_gate(r"SNOT",1)
qc.add_gate(r'RZ',1,None, np.pi/2, r'pi/2')

qc.png
