#/usr/env python

#import libs
import numpy as np
# numpy triu_indices to return the indices for the upper-triangle of an (n,m) array

def linear_to_ij(n,k):
    """
    params: n - size of matrix ; k - lineear triu index 
    """
    i = n - 2 - int( np.sqrt(-8*k + 4*n*(n-1)-7)/2. - 0.5 )
    j = k + i + 1 - n*(n-1)/2 + (n-i)*( (n-i)-1 )/2 
    # to assert that the calculated indices are truth
    assert np.triu_indices(n, k = 1)[0][k] == i
    assert np.triu_indices(n, k = 1)[1][k] == j
    return int(i),int(j)

def ij_to_linear(n,i,j):
    """
    params: n - size of matrix ; i,j : indices 
    """
    k = ( n*(n-1)/2 ) - (n-i)*( (n-i)-1 )/2 + j - i - 1
    # to assert that the calculated indices are truth
    assert np.triu_indices(n, k = 1)[0][k] == i
    assert np.triu_indices(n, k = 1)[1][k] == j
    return int(k)

