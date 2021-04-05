import numpy as np
def get_one_hot(length:int,index:int):
    b = np.zeros(length)
    b[index] = 1
    return b