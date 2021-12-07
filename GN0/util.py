import numpy as np
def get_one_hot(length:int,index:int,dtype=np.float32):
    """Returns a zero vector with one entry set to one
    
    Args:
        index: The index of the entry to set to one
        length: The lenght of the output vector
        dtype: The dtype of the desired vector

    Returns:
        A numpy array of one-hot format
    """
    b = np.zeros(length,dtype=np.float32)
    b[index] = 1
    return b
def get_alternating(length:int,even,odd,dtype=np.float32):
    """Get an array with alternating values

    Args:
        length: The length of the desired array
        even: The value to put at even indices of the array (Assuming 0 as starting index)
        odd: The value to put at odd indices of the array
        dtype: The dtype of the desired array

    Returns:
        A numpy array with alternating values
    """
    out = np.empty(length,dtype=dtype)
    out[::2] = even
    out[1::2] = odd
    return out