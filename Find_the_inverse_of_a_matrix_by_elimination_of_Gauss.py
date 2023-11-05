from Solve_a_system_by_the_elimination_of_Gauss import Solve
import numpy as np
def Inverse(A):
    n = A.shape[0]
    inv_A = np.empty((n,n))
    b = np.array([0]*n)
    for i in range(n):
        b[i] = 1
        inv_A[:,i] = Solve(A,b)
        b[i] = 0
        
    return inv_A
if __name__ == "__main__" :
    A = np.array([[25,5,1],
                [64,8,1],
                [144,12,1]])

    invA = Inverse(A)
    print("A@invA = \n",np.around(A@invA))

