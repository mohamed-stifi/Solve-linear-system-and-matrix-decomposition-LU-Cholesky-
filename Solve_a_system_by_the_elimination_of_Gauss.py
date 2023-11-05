import numpy as np
from time import time
import matplotlib.pyplot as plt
np.random.seed(42)
def for_Elimination(A,b):
    n = A.shape[0]
    arr = np.zeros((n, n+1))
    arr[:,:n] = A.copy()
    arr[:,n] = b.copy() 
    for i in range(n-1):
        p_ind = i + np.argmax(np.abs(arr[i:,i]))
        arr[i,i:], arr[p_ind,i:] = arr[p_ind,i:].copy(),arr[i,i:].copy()
        pivot = arr[i,i]
        if pivot == 0 : print("le systeme n admet pas de solution unique ")
        for k in range(i+1, n):
            m = arr[k,i]/pivot
            arr[k,i:] -= m*arr[i, i:]
    return arr[:,:n], arr[:,n]

def back_Substitution(A,b):
    n = A.shape[0]
    X = np.empty(n)
    X[-1] = b[-1]/A[-1,-1]
    for i in range(-2, -(n+1), -1):
        X[i] = (b[i] - np.dot(A[i,i+1:], X[i+1:]))/A[i,i]
    
    return X

def Solve(A,b):
    """
     Solve a system S of linear equations A x = b defined \
        by the matrix A and the vector b
    """
    a, bb = for_Elimination(A,b)
    return back_Substitution(a, bb)
if __name__ == "__main__":
    n = 100
    A = np.random.randn(n,n)

    b = np.random.randn(n)

    names = ["My method", "numpy method"]
    t = []
    # -------------------------------------------------------
    t1 =time(); X = Solve(A,b);t.append(time()-t1)
    t1 =time(); x = np.linalg.solve(A,b);t.append(time()-t1)
    # -------------------------------------------------------
    plt.bar(names,t, label = "n = "+str(n))
    plt.legend()
    plt.show()
    # -------------------------------------------------------
    print(list(zip(names,t)))
    print(f"the percentage of X == x is : {np.sum(np.around(X,9) == np.around(x,9))*100/n} % ")



