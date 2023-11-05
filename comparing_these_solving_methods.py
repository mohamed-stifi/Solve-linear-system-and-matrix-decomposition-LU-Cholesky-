from Find_the_inverse_of_a_matrix_by_elimination_of_Gauss import Inverse as Gauss_Inv
from Choleski_decomposition import Choleski_decomposition
from Inverse_by_Choleski_decomposition import Inverse_by_choleski as Choleski_Inv
from Solve_a_system_by_LU_decomposition import Solve as LU_solve
from Solve_a_system_by_the_elimination_of_Gauss import Solve as Gauss_solve
from Implementation_for_the_LU_decomposition import LU, PLU
from LU_Inverse import LU_Inverse as LU_Inv
from time import time 
import matplotlib.pyplot as plt
import numpy as np 
np.random.seed = 42
n = 10
names = ["LU_inv", "PLU_inv", "numpy_inv", "Gauss_inv",
        "Choleski_inv", "LU_solve",
        "PLU_solve", "Gauss_solve", "numpy_solve"]
times = []

# A = np.array([[2, -1, 0],[-1, 2, -1],[0,-1,2]], dtype= np.double)
A = np.random.randn(n,n)
b = np.random.randn(n)
# -----------------------------------------------------
t = time()
l,u = LU(A)
LU_Inv(l,u)
times.append(time()- t)
# -----------------------------------------------------
t = time()
p,l,u = PLU(A)
LU_Inv(l,u,p)
times.append(time()- t)
# -----------------------------------------------------
t = time()
np.linalg.inv(A)
times.append(time()- t)
# -----------------------------------------------------
t = time()
Gauss_Inv(A)
times.append(time()- t)
# -----------------------------------------------------
t = time()
l = Choleski_decomposition(A)
Choleski_Inv(l)
times.append(time()- t)
# -----------------------------------------------------
t = time()
l,u = LU(A)
LU_solve(l,u,b)
times.append(time()- t)
# -----------------------------------------------------
t = time()
p,l,u = PLU(A)
LU_solve(l,u, np.dot(p,b))
times.append(time()- t)
# -----------------------------------------------------
t = time()
Gauss_solve(A,b)
times.append(time()- t)
# -----------------------------------------------------
t = time()
np.linalg.solve(A,b)
times.append(time()- t)
# -----------------------------------------------------

plt.bar(names[:-4], times[:-4], alpha = 0.6 )
plt.bar(names[-4:], times[-4:], alpha = 0.6 )
plt.title("runing time of these method")
plt.xlabel("methods names")
plt.ylabel("time (s)")
plt.show()
