import numpy as np

def Choleski_decomposition(A):
    L = np.zeros(A.shape, dtype= np.double)
    n = A.shape[0]
    for k in range(n):
        L[k,k] = np.sqrt(A[k,k] - np.dot(L[k,:k], L[k,:k]))

        for i in range(k+1, n):
            L[i,k] = ( A[i,k] - np.dot(L[i,:k], L[:k,k]) )/L[k,k]
        
    return L
if __name__ == "__main__":

    A = np.array([[2, -1, 0],[-1, 2, -1],[0,-1,2]], dtype= np.double)


    L = Choleski_decomposition(A)
    l = np.linalg.cholesky(A)
    print("Choleski_decomposition L = \n",L)
    print("numpy choleski l = \n",l)