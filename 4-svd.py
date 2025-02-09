import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from skimage import data
from skimage.io import imread

A = data.moon()


print(type(A))
print('Shape of A: ', A.shape)


plt.title("Immagine originale")
plt.imshow(A, cmap='gray')
plt.show()

A_p = np.zeros(A.shape)
p_max = 15

U, s , Vh = scipy.linalg.svd(A)


for i in range(p_max):
    p = i + 1
    ui = U[:, i]
    vi = Vh[i, :]
    A_p = A_p + s[i] * np.outer(ui, vi)

print("A_p = \n", A_p)
print('\n')
plt.imshow(A_p, cmap = 'gray')
plt.title("Immagine costruita come somma di diadi")
plt.show()

err_rel = np.linalg.norm(A-A_p, 2)/np.linalg.norm(A,2)

m = U.size
n = Vh.size
c = min(m, n) / p_max - 1

print('L\'errore relativo della ricostruzione di A è', err_rel)
print('Il fattore di compressione è c=', c)
print('\n')


plt.figure(figsize=(20, 10))

fig1 = plt.subplot(1, 2, 1)
fig1.imshow(A, cmap='gray')
plt.title('Immagine originale')

fig2 = plt.subplot(1, 2, 2)
fig2.imshow(A_p, cmap='gray')
plt.title('Immagine ricostruita con p =' + str(p_max))

plt.show()

# calcolare e plottare al variare di p

s_v = np.arange(2, 20, 2)
err_rel = np.zeros(np.size(s_v))
c = np.zeros(np.size(s_v))


j = 0
for p_max in s_v:      # P_max indica il numero di diadi che comporranno la matrice.
                        # una diade é una matrice [n, 1] e NON é un vettore.

    A_p = np.zeros(A.shape)
    U, s, VT = scipy.linalg.svd(A)

    for i in range(p_max):
      ui = U[:, i]
      vi = VT[i, :]
      A_p = A_p + (np.outer(ui, vi) * s[i])

    m = U.size    # Servono per calcolare il fattore di compressione.
    n = VT.size
    c[j] = min(m, n) / p_max - 1

    err_rel[j] = np.linalg.norm(A - A_p, 2) / np.linalg.norm(A, 2)


    print('Errore relativo della ricostruzione di A = ', err_rel[j])
    print('Fattore di compressione è c =', c[j])
    print('\n')
    j = j + 1

plt.plot(s_v, err_rel, '-m*')
plt.title('Errore relativo sul numero di diadi p')
plt.xlabel('Dimensione di p')
plt.ylabel('Errore relativo')
plt.show()

plt.plot(s_v, c, '-g*')  # É normale che il fattore di compressione diminuisca col procedere delle iterazioni
plt.title('Fattore di compressione sul numero di diadi p')
plt.xlabel('Dimensione di p')
plt.ylabel('Compressione')
plt.show()
