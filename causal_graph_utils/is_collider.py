
def is_collider (X, Y, Z, matrix):
    '''
    Check if Y is a collider in the triplet X -Y - Z
    Args:
        X (int): node X
        Y (int): node Y
        Z (int): node Z
        matrix (numpy array): the matrix of the causal graph

    Returns:
        is_collider (bool) : True (if X *->Y <-*Z ),  False otherwise
    '''

    is_collider = False
    if matrix[X,Y] == 2 and matrix[Z, Y] == 2:
        is_collider = True

    return is_collider