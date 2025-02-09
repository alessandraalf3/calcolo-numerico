import numpy as np
import scipy
import scipy.linalg

n = np.random.randint(10, 1001)

A = np.random.randn(n, n)
x = np.ones((n, 1))
b = A @ x


print('x.shape: ', x.shape, '\n' )

print('b.shape: ', b.shape, '\n' )

print('A.shape: ', A.shape, '\n' )

print('K(A)=', np.linalg.cond(A), '\n')

lu, piv = scipy.linalg.lu_factor(A)

print('lu',lu,'\n')
print('piv',piv,'\n')


my_x = scipy.linalg.lu_solve((lu, piv), b)

print('my_x = \n', my_x)
