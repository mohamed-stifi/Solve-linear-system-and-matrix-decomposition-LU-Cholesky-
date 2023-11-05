from Implementation_for_the_LU_decomposition import LU, PLU
import numpy as np

def Solve_Lz_b(L,b):
    n = L.shape[0]
    Z = np.zeros(n, dtype= np.double)
    Z[0] = b[0]/L[0,0]
    for i in range(1,n):
        Z[i] = (b[i] - np.dot(L[i,:i], Z[:i]))/L[i,i]
    return Z

def Solve_Ux_z(U,z):
    n = U.shape[0]
    X = np.zeros(n, dtype= np.double)
    X[-1] = z[-1]/U[-1,-1]
    for i in range(n-2, -1, -1):
        X[i] = (z[i] - np.dot(U[i,i+1:], X[i+1:]) )/U[i,i]
    return X

def Solve(L,U ,b):
    z = Solve_Lz_b(L,b)
    x = Solve_Ux_z(U,z)
    return x
if __name__ == "__main__":
    A = np.array([[20,4,-2],
                [4,9,-3],
                [-2,-3,7]], dtype=np.double)
    b = np.array([2,5,10])

    L,U = LU(A)
    Xlu = Solve(L,U,b)
    print("x by LU = ",Xlu)
    P,L,U = PLU(A)
    Xlu = Solve(L,U,np.dot(P,b))
    print("x by PLU = ",Xlu)
    x = np.linalg.solve(A,b)
    print("x  = ",x)


    