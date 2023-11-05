from Choleski_decomposition import Choleski_decomposition
from Solve_a_system_by_LU_decomposition import Solve_Lz_b as for_sub
from Solve_a_system_by_LU_decomposition import Solve_Ux_z as back_sub
import numpy as np

def Inverse_by_choleski(L):
    n = L.shape[0]
    Inv = np.empty((n,n), dtype=np.double)
    x = np.empty(n, dtype=np.double)
    b = np.zeros(n, dtype=np.double)
    for i in range(n):
        b[i] = 1
        # Solve the equation L * X = b for X using forward substitution
        x = for_sub(L, b)
        # Solve the equation L^T * Y = X for Y using back substitution
        x = back_sub(L.T, x)
        # Set the i-th column of the inverse matrix
        Inv[:,i] = x
        b[i] = 0
    return Inv

    return Inv
if __name__ == "__main__":
    A = np.array([[2, -1, 0],[-1, 2, -1],[0,-1,2]], dtype= np.double)

    L = Choleski_decomposition(A)
    Choleski_InvA = Inverse_by_choleski(L)
    numpy_invA = np.linalg.inv(A)
    print(f"Inverse of A by choleski decomposition is \n {Choleski_InvA}")
    print(f"Inverse of A by numpy is \n {numpy_invA}")

