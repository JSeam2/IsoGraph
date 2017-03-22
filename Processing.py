
"""
Processes the graph data to feed in to a neural network

Will attempt to address 10 nodes first
"""

import sqlite3
import numpy as np


def openDB(path ='./graph.db'):
    """
    1) connects to DB location
    2) extracts data for use in neural network
    3) Converts into np array for use in Tensorflow 
    """
    sqlite3.connect(path)

