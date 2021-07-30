"""
Utility functions for Mini Project 2
Ali Murad
"""
import numpy as np

def binary_converter(y):
    """
    Convert decimal classes to binary
    """
    decimal_to_binary = {0: '0000', 1: '0001', 2: '0010', 3: '0011', 4: '0100', 5: '0101', 6: '0110', 
                  7: '0111', 8: '1000', 9: '1001'}

    ecoc_labels = np.vectorize(decimal_to_binary.get)(y)
    Y_1 = np.array([int(e[0]) for e in ecoc_labels])
    Y_2 = np.array([int(e[1]) for e in ecoc_labels])
    Y_3 = np.array([int(e[2]) for e in ecoc_labels])
    Y_4 = np.array([int(e[3]) for e in ecoc_labels])
        
    return Y_1, Y_2, Y_3, Y_4

def decimal_converter(y):
    """
    Binary to decimal
    """
    binary_to_decimal = {'0000': 0, '0001': 1, '0010':2, '0011': 3, '0100': 4, '0101': 5, '0110': 6, 
                  '0111': 7, '1000': 8, '1001': 9, '1010': 9, '1011': 9, '1100': 9, '1101': 9, 
                  '1110': 9, '1111': 9}

    converted_labels = np.vectorize(binary_to_decimal.get)(y)
    
    return converted_labels