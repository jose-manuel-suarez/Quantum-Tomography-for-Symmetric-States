# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 14:46:25 2021

@author: Holik
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
import numpy as np
from scipy import linalg as la
from itertools import permutations
import itertools

import sympy

import scipy.sparse as sp
ugh = []

from scipy.sparse import csr_matrix, lil_matrix


def von_neumann_entropy(rho):
    R = rho * (la.logm(rho) / la.logm(np.matrix([[2]])))
    S = -np.matrix.trace(R)
    return S


def von_neumann_entropyEig(rho):
    EV = la.eigvals(rho)
    # Drop zero eigenvalues so that log2 is defined
    my_list = [x for x in EV.tolist() if x.real >= 0]
    EV = np.array(my_list)
    log2_EV = np.matrix(np.log2(EV))
    EV = np.matrix(EV)
    S = -np.dot(EV, log2_EV.H)
    T = S[0, 0]
    T = T.astype(float)
    return T


def fidelity(A, B):
    sqrtmA = la.sqrtm(A)
    NN = sqrtmA @ B @ sqrtmA
    eig_vals = la.eig(NN)[0]
    positive_list = [x for x in eig_vals.tolist() if x.real >= 0]
    return float(np.real(np.sqrt(eig_vals).sum()) ** 2)


# Producto tensorial de todos contra todos en dos listas.


def KroneckerTwoSets(L1, L2):
    P = []
    for i in range(len(L1)):
        for j in range(len(L2)):
            P.append(np.kron(L1[i], L2[j]))
    return P


def KroneckerNSets(L, N):
    R = list(itertools.repeat(L, N - 1))
    result = L
    for x in R:
        result = KroneckerTwoSets(result, x)
    return result


def KroneckerTwoSets_SP(L1, L2):
    P = []
    for i in range(len(L1)):
        for j in range(len(L2)):
            L1[i] = csr_matrix(L1[i])
            L2[j] = csr_matrix(L2[j])
            P.append(sp.kron(L1[i], L2[j]))
    return P


def KroneckerNSets_SP(L, N):
    R = list(itertools.repeat(L, N - 1))
    result = L
    for x in R:
        result = KroneckerTwoSets_SP(result, x)
    return result


# Producto matricial todos contra todos en una lista de matrices.


def MatrixProductTwoSets(L1, L2):
    P = []
    for i in range(len(L1)):
        for j in range(len(L2)):
            P.append(np.matmul(L1[i], L2[j]))
    return P


def MatrixProducNSets(L, N):
    R = list(itertools.repeat(L, N - 1))
    result = L
    for x in R:
        result = MatrixProductTwoSets(result, x)
    return result


# Esto es para hacer el producto tensorial de una lista de matrices:


def KroneckerMultiplyList(myList):
    # Multiply elements one by one
    result = [[1]]
    for x in myList:
        result = np.kron(result, x)
    return result


def KroneckerMultiplyList_SP(myList):
    # Multiply elements one by one
    result = csr_matrix([[1]])
    for x in myList:
        x = csr_matrix(x)
        result = sp.kron(result, x)
    return result


# Esto es para armar listas de un cierto largo con Paulis (para simetrizar después).


def Lists(L, N):
    R = list(itertools.repeat(L, N))
    M = []
    for i in itertools.product(*R):
        M.append(i)
    return M


# Todo esto es para simetrizar matrices:


def Symmetrizer(L):
    Perm = list(permutations(L))
    symm = sum(
        (1 / np.math.factorial(len(L))) * KroneckerMultiplyList_SP(a) for a in Perm
    )
    return symm


def differentSquares(L):
    B = tuple(L)
    A = set()  # Create a set for adding unique 2x2 matrices
    for i in range(len(L)):
        temp = L[i]
        A.add(temp)  # Add the matrix. It will not add any duplicates
    return A  # Returns 6 for the given example


# Para armar la base canónica de matrices de NxN (ojo que N es el tamaño de la matriz en este contexto, habría que cambiar la notación).


def f(size, indexA, indexB):
    dimensions = (size, size)
    arr = np.zeros(dimensions)
    arr[indexA, indexB] = 1.0
    return arr


def CanonicalBasis(N):
    B = []
    for i in range(N):
        for j in range(N):
            temp = f(N, i, j)
            B.append(temp)
    return B


# Acá armo una base de las Hermíticas. Tienen n matrices con un uno en la diagonal y todos ceros,
# n(n−1)/2 matrices con un par de unos en los off diagonal, y n(n−1)/2 matrices con i y -i en los off diagonal.


def h(size, indexA, indexB):
    dimensions = (size, size)
    arr = np.zeros(dimensions)
    arr[indexA, indexB] = 1.0
    arr[indexB, indexA] = 1.0
    return arr


def hc(size, indexA, indexB):
    dimensions = (size, size)
    arr = np.zeros(dimensions, dtype=complex)
    arr[indexA, indexB] = -1j
    arr[indexB, indexA] = 1j
    return arr


def CanonicalBasisH(N):
    B = []
    for i in range(N):
        for j in range(N):
            if i == j:
                temp = f(N, i, j)
                B.append(temp)
            elif i < j:
                temp1 = h(N, i, j)
                B.append(temp1)
            elif j < i:
                temp2 = hc(N, i, j)
                B.append(temp2)
    return B

# Acá defino Gram Schmidt para un conjunto arbitrario de matrices (con el producto escalar que da la traza).

# def gram_schmidt(vectors):
#     basis = []
#     for v in vectors:
#         w = v - sum( (np.trace(np.matmul(v,b.conj().T)))*b  for b in basis )
#         if np.sqrt(np.trace(np.matmul(w,w.conj().T))) > 1e-18:
#             basis.append(w/np.sqrt(np.trace(np.matmul(w,w.conj().T))))
#         # else:
#         #     print(np.sqrt(np.trace(np.matmul(w,w.conj().T))))
#     return np.array(basis)


def Gram_Schmidt(vectors, n, search_LI = False):
    '''
    Gram Schmidt para un conjunto arbitrario de matrices (con el producto escalar que da la traza).
    Devuelve un array de Numpy cuyos elementos son las matrices de la base ortog
    (opcional) search_LI: devuelve conjunto LI haciendo Gram-Schmidt y descartando vectores nulos
    '''
    basis = []
    LI_basis = []
    counter = 0
    l = len(vectors)
    loop = n
    for v in vectors:
        Products = [ np.trace(np.matmul(v,b.conj().T)) for b in basis ]
        Relevant = [ [Products[j],basis[j]] for j in range(len(basis)) if Products[j]!= 0 ]
        w = v - sum( Relevant[k][0]*Relevant[k][1] for k in range(len(Relevant)) )
        counter = counter + 1
        print(f"Va por el vector: {counter}")
        print(f"Va por el loop: {loop}")
        print(f"Faltan: {l-counter}")
        if np.abs(np.trace(np.matmul(w,w.conj().T))) > 1e-14:      
            basis.append(w/np.sqrt(np.trace(np.matmul(w,w.conj().T))))
            if search_LI:
                LI_basis.append(v)
    if search_LI:
        return LI_basis
    else:            
        return np.array(basis)

# def Gram_Schmidt(vectors, search_LI=False):
#     """
#     Gram Schmidt para un conjunto arbitrario de matrices (con el producto escalar que da la traza).
#     Devuelve un array de Numpy cuyos elementos son las matrices de la base ortog
#     (opcional) search_LI: devuelve conjunto LI haciendo Gram-Schmidt y descartando vectores nulos
#     """
#     basis = []
#     LI_basis = []
#     counter = 0
#     l = len(vectors)
#     for v in vectors:
#         u = sum( ((v @ b.conj().T).diagonal()).sum() * b for b in basis)
#         w = v - u
#         counter = counter + 1
#         print(f"Va por el vector: {counter}")
#         print(f"Faltan: {l-counter}")
#         if ((w @ w.conj().T).diagonal()).sum() > 1e-5:
#             basis.append(w/np.sqrt(((w @ w.conj().T).diagonal()).sum()))
#             if search_LI:
#                 LI_basis.append(v)
#     if search_LI:
#         return LI_basis
#     else:
#         return np.array(basis)


# Norma de w: /np.sqrt(np.array((w@w.conj().T).diagonal()).sum())
# Esto agarra un conjunto de matrices de nxn, les anexa la base canónica de las matrices de nxn HERMÍTICAS, y a eso le saca un subconjunto Linealmente Independiente.
# Es decir, me da un conjunto de generadores de las matrices de nxn, que contiene a los vectores del conjunto original (le saca los que son LD, y completa con cositos de la canónica de las Hermíticas).


def Generate_LI(L, n):
    L2 = L + CanonicalBasisH(n)
    W = list(L2[i].flatten() for i in range(len(L2)))
    inds2 = sympy.Matrix(W).T.rref()  # to check the rows you need to transpose!
    my_tuple2 = list(inds2[1])
    LISet = list(L2[i] for i in my_tuple2)
    return LISet

# Producto escalar con la traza y su norma asociada.


def TSP(A, B):
    P = np.trace(np.matmul(A, B.conj().T))
    return P


def Norm(A):
    Q = np.sqrt(np.trace(np.matmul(A, A.conj().T)))
    return Q


# Permutacion pi aplicada a una lista.

def Perm(pi, L):
    temp = [L[i] for i in pi]
    Wtemp = KroneckerMultiplyList(temp)
    return Wtemp


def Perm_SP(pi, L):
    temp = [L[i] for i in pi]
    Wtemp = KroneckerMultiplyList_SP(temp)
    return Wtemp


# Matriz de permutación asociada a la permutación Pi.


def P(pi, N):
    Zero = np.array([[1], [0]])
    One = np.array([[0], [1]])
    B = [Zero, One]
    W = Lists(B, N)
    PpiN = sum(np.outer(Perm(pi, w), KroneckerMultiplyList(w)) for w in W)
    return PpiN


def P_SP(pi, N):
    Zero = np.array([[1], [0]])
    One = np.array([[0], [1]])
    B = [Zero, One]
    W = Lists(B, N)
    PpiN = sum(np.outer(Perm_SP(pi, w), KroneckerMultiplyList_SP(w)) for w in W)
    return PpiN
