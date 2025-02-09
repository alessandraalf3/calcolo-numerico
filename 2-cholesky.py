import numpy as np
import scipy
import scipy.linalg
from numpy.linalg import cholesky

n = np.random.randint(10, 101)

A = A = np.diag(9 * np.ones(n)) + np.diag(-4 * np.ones(n-1), k=1) + np.diag(-4 * np.ones(n-1), k=-1)
x_True = np.ones((n, 1))
b = A @ x_True


print('A.shape: ', A.shape, '\n' )

print('K(A)=', np.linalg.cond(A), '\n')

L = cholesky(A)
my_x = np.linalg.solve(L.T, np.linalg.solve(L, b))
print(f"Soluzione calcolata con la fattorizzazione di Cholesky: \n {my_x}")

