import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

m = 8
x = np.linspace(-1, 1, m)
y = 1/(1+25*x**2)
degree_pol = np.array((1, 2, 3, 5, 7))

for n in degree_pol:
    A = np.zeros((m, n+1))
    for i in range(n+1):
        A[:, i] = x**i

    U, s , Vh = scipy.linalg.svd(A)
    x_svd = np.zeros(n+1)

    ATA = np.matmul(A.T, A)
    ATy = np.matmul(A.T, y)

    lu, piv = scipy.linalg.lu_factor(ATA)
    x_eqnorm = scipy.linalg.lu_solve((lu, piv), ATy)
    print("Soluzione equazioni normali = ", x_eqnorm, '\n')

    res_eqnorm = np.linalg.norm(A @ x_eqnorm - y)
    print("Errore equazioni normali = ", res_eqnorm, '\n')

    for i in range(n+1):
        ui = U[:, i]
        vi = Vh[i, :]

        x_svd = x_svd + (np.dot(ui.T, y) *vi) / s[i]

    print("Soluzione SVD = ", x_svd, '\n')

    res_svd = np.linalg.norm(A @ x_svd - y)
    print("Errore svd = ", res_svd, '\n')

    m_plot = 100
    x_plot = np.linspace(x[0], x[-1], m_plot)
    A_plot = np.zeros((m_plot, n+1))

    for i in range(n+1):
        A_plot[:, i] = x_plot ** i

    y_eqnorm = A_plot @ x_eqnorm


    fig1 = plt.subplot(1, 2, 1)
    plt.title(f"Grado {n} con equazioni normali")
    plt.plot(x, y, '*r')
    plt.plot(x_plot, y_eqnorm, 'm')
    plt.grid()

    plt.show()


    y_svd = A_plot @ x_svd

    fig2 = plt.subplot(1, 2, 2)
    plt.title(f"Grado {n} con SVD")
    plt.plot(x, y, '*r')
    plt.plot(x_plot, y_svd, 'm')
    plt.grid()

    plt.show()
