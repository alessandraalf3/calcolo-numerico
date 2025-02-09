import numpy as np
from scipy.linalg import hilbert
from numpy.linalg import cholesky

n = np.random.randint(2, 16)

A = A = hilbert(n)
x_True = np.ones((n, 1))
b = np.dot(A, x_True)


print('x_True.shape: ', x_True.shape, '\n' )

print('b.shape: ', b.shape, '\n' )

print('A.shape: ', A.shape, '\n' )

print('K(A)=', np.linalg.cond(A), '\n')

L = cholesky(A)
my_x = np.linalg.solve(L.T, np.linalg.solve(L, b))
print(f"Soluzione calcolata con la fattorizzazione di Cholesky: \n {my_x}")
