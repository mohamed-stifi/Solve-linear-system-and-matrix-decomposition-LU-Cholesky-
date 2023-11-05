from Solve_a_system_by_LU_decomposition import Solve_Lz_b as for_sub
from Solve_a_system_by_LU_decomposition import Solve_Ux_z as back_sub
from Implementation_for_the_LU_decomposition import LU, PLU

import numpy as np

def LU_Inverse(L,U, P = None):
    n = L.shape[0]
    Inv = np.empty((n,n), dtype=np.double)
    x = np.empty(n, dtype=np.double)
    b = np.zeros(n, dtype=np.double)
    for i in range(n):
        b[i] = 1
        # Solve the equation L * X = b for X using forward substitution
        if P is not None:
            x = for_sub(L,np.dot(P,b))
        else :
            x = for_sub(L, b)
        # Solve the equation L^T * Y = X for Y using back substitution
        x = back_sub(U, x)
        # Set the i-th column of the inverse matrix
        Inv[:,i] = x
        b[i] = 0
    return Inv

if __name__ == "__main__":
    A = np.array([[2, -1, 0],[-1, 2, -1],[0,-1,2]], dtype= np.double)

    L, U = LU(A)

    LU_InvA = LU_Inverse(L,U)
    P,L,U = PLU(A)
    PLU_InvA = LU_Inverse(L,U, P)
    numpy_invA = np.linalg.inv(A)
    print(f"Inverse of A by LU decomposition is \n {LU_InvA}")
    print(f"Inverse of A by PLU decomposition is \n {PLU_InvA}")
    print(f"Inverse of A by numpy is \n {numpy_invA}")


