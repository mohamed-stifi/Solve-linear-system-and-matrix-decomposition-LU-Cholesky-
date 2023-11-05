import numpy as np
import time as t
import scipy.linalg
np.random.seed(42)
# Define the matrix
n = 50
A = np.random.randn(n,n)
# ----------| implementation for the LU decomposition|-------
def LU(A, ameliorer  = False) :
    n = len(A)
    U = A.copy()
    L = np.eye(n)
    for i in range(n-1):

        pivot =  U[i,i]
        if pivot == 0 : print("pivot = 0") ; return L,U

        for k in range(i+1, n):
            L[k,i] = U[k,i]/pivot
            U[k] = U[k] - L[k,i]*U[i]
    return L,U

def PLU(A):
    n = A.shape[0]
    U = A.copy()
    L = np.eye(n, dtype= np.double)
    P = np.eye(n, dtype= np.double)

    for i in range(n):
        for k in range(i,n):
            if ~np.isclose(U[i,i], 0) :
                break
            U[[k,k+1]] = U[[k+1, k]]
            P[[k,k+1]] = P[[k+1, k]]

        for k in range(i+1, n):
            L[k,i] = U[k,i]/U[i,i]
            U[k] = U[k] - L[k,i]*U[i]
    return P,L,U



if __name__ == "__main__":
    # ------------------------------------------------------------
    t1 = t.time() ; L,U = LU(A) ;t1 = t.time() - t1
    t2 = t.time();p, l,u = scipy.linalg.lu(A);t2 = t.time() - t2
    t3 = t.time();p, pL, pU = PLU(A) ; t3 = t.time() - t3
    # ------------------------------------------------------------
    print("my LU time : ",np.around(t1,10))
    print("my PLU time : ",np.around(t3,10))
    # ------------------------------------------------------------
    print("numpy lu time : ",np.around(t2,10))
    # ------------------------------------------------------------
    print(f"A is a matrix of shape {n}x{n}")
    print(f"L == l : {np.sum(np.around(l) == np.around(L))*100/len(l)**2}%")
    print(f"U == u : {np.sum(np.around(u) == np.around(U))*100/len(l)**2}%")
    print(f"A == L*U : {np.sum(np.around(A) == np.around(L@U))*100/len(l)**2}%")
    print(f"A == l*u : {np.sum(np.around(A) == np.around(l@u))*100/len(l)**2}%")
    print(f"L@U == l*u : {np.sum(np.around(L@U) == np.around(l@u))*100/len(l)**2}%")

    print(f"pL == l : {np.sum(np.around(l) == np.around(pL))*100/len(l)**2}%")
    print(f"pU == u : {np.sum(np.around(u) == np.around(pU))*100/len(l)**2}%")
    print(f"A == pL*pU : {np.sum(np.around(A) == np.around(pL@pU))*100/len(l)**2}%")
    print(f"pL@pU == l*u : {np.sum(np.around(pL@pU) == np.around(l@u))*100/len(l)**2}%")
    
