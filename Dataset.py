# -*- coding: utf-8 -*-
"""
Created on Sun May  2 14:48:55 2021

@author: Yuanhang Zhang
"""

import numpy as np
import itertools
from scipy import sparse
from scipy.sparse.linalg import eigsh
import torch

# device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X = sparse.csr_matrix(np.array([[0, 1], [1, 0]], dtype=np.float64))
Y = sparse.csr_matrix(np.array([[0, -1j], [1j, 0]], dtype=np.complex128))
Z = sparse.csr_matrix(np.array([[1, 0], [0, -1]], dtype=np.float64))
I = sparse.csr_matrix(np.array([[1, 0], [0, 1]], dtype=np.float64))
Sp = sparse.csr_matrix(np.array([[0, 1], [0, 0]], dtype=np.float64))
Sm = sparse.csr_matrix(np.array([[0, 0], [1, 0]], dtype=np.float64))


def dec2bin(x, bits):
    '''
    credit to Tiana
    https://stackoverflow.com/questions/55918468/convert-integer-to-pytorch-tensor-of-binary-bits
    '''
    # mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).to(torch.get_default_dtype())


def bin2dec(b, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
    return torch.sum(mask * b, -1)


class Hamiltonian():
    def __init__(self, n):
        self.n = n

    def Eloc(self, v):
        pass

    def full_H(self):
        pass

    def calc_E_ground(self):
        try:
            H = self.H
        except AttributeError:
            H = self.full_H()
        [self.E_ground, self.psi_ground] = eigsh(H, k=1, which='SA')
        self.E_ground = self.E_ground[0]
        self.psi_ground = self.psi_ground[:, 0]
        return self.E_ground


class Ising2D(Hamiltonian):
    # triangular
    # modify self.connections for arbitrary connecction
    def __init__(self, nx, ny, J, h):
        n = nx * ny
        super(Ising2D, self).__init__(n)
        self.nx = nx
        self.ny = ny
        self.n = n
        self.J = J
        self.h = h

        # triangular, periodic
        self.connections = []
        for i in range(nx):
            for j in range(ny):
                self.connections.append([self.index(i, j), self.index(i + 1, j)])
                self.connections.append([self.index(i, j), self.index(i, j + 1)])
                self.connections.append([self.index(i, j), self.index(i + 1, j + 1)])

        self.connections = np.array(self.connections, dtype=int)

    def index(self, x, y):
        # periodic boundary conditions
        return (x % self.nx) * self.ny + (y % self.ny)

    def full_H(self):
        self.H = sparse.csr_matrix((2 ** self.n, 2 ** self.n), dtype=np.float64)
        for conn in self.connections:
            JZZ = 1
            for i in range(self.nx):
                for j in range(self.ny):
                    idx = self.index(i, j)
                    if idx == conn[0]:
                        JZZ = sparse.kron(JZZ, Z, format='csr')
                    elif idx == conn[1]:
                        JZZ = sparse.kron(JZZ, Z, format='csr')
                    else:
                        JZZ = sparse.kron(JZZ, I, format='csr')
            self.H = self.H + self.J * JZZ
        for i in range(self.n):
            hX = 1
            for j in range(self.n):
                if i == j:
                    hX = sparse.kron(hX, X, format='csr')
                else:
                    hX = sparse.kron(hX, I, format='csr')
            self.H = self.H - self.h * hX
        return self.H

    def Eloc(self, v, rbm):
        connections = torch.tensor(self.connections, dtype=torch.int64).T
        batch = v.shape[0]
        batch_idx = torch.arange(batch).reshape(batch, 1)
        v0 = v[batch_idx, connections[0]]
        v1 = v[batch_idx, connections[1]]
        E = self.J * (v0 * v1).sum(dim=1)

        if rbm.is_amp_phase:
            theta, thetai = rbm.calc_theta(v)
            vW = torch.einsum('bn, nm->nbm', v, rbm.W)
            vWi = torch.einsum('bn, nm->nbm', v, rbm.Wi)
            temp = 0.5 * (rbm.logcosh(theta - 2 * vW) - rbm.logcosh(theta)) \
                   + 0.5j * (rbm.logcosh(thetai - 2 * vWi) - rbm.logcosh(thetai))
            temp = (temp.sum(dim=2).T - v * rbm.a - 1j * v * rbm.ai).exp()
        else:
            theta = rbm.calc_theta(v)
            vW = torch.einsum('bn, nm->nbm', v, rbm.W)
            temp = 0.5 * (rbm.logcosh(theta - 2 * vW) - rbm.logcosh(theta))
            temp = (temp.sum(dim=2).T - v * rbm.a).exp()

        E = E - self.h * temp.sum(dim=1)
        return E


class Dataset():
    def __init__(self, n, sample_size):
        self.n = n
        self.sample_size = sample_size

    def sample_weighting(self, samples):
        samples, sample_weight = torch.unique(samples, dim=0, return_counts=True)
        sample_weight = sample_weight / torch.sum(sample_weight)
        return samples, sample_weight

    def sample(self, sample_size=-1):
        if sample_size < 0:
            sample_size = self.sample_size
        idx = torch.multinomial(self.data_weight, sample_size, replacement=True)
        samples = self.data[idx, :]
        return samples

    def dec2bin(self, x, bits):
        '''
        credit to Tiana
        https://stackoverflow.com/questions/55918468/convert-integer-to-pytorch-tensor-of-binary-bits
        '''
        # mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
        mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
        return x.unsqueeze(-1).bitwise_and(mask).ne(0).to(torch.get_default_dtype())

    def bin2dec(self, b, bits):
        mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
        return torch.sum(mask * b, -1)


class W_state(Dataset):
    def __init__(self, n, sample_size, n_measure):
        super(W_state, self).__init__(n, sample_size)
        self.data = torch.eye(n, device=device)
        self.data_weight = torch.ones(n, device=device) / n
        self.n_measure = n_measure
        self.measure_Z(n_measure)
        self.calc_basis_W()

    def full_psi(self):
        self.psi = torch.zeros(2 ** self.n, device=device)
        idx = 2 ** torch.arange(self.n, device=device)
        self.psi[idx] = 1 / np.sqrt(self.n)
        return self.psi

    def calc_basis_W(self):
        n = self.n
        self.basis_W = torch.eye(n)
        self.basis_extended = torch.zeros([int(1 + n + n * (n - 1) / 2), n])  # states with 0, 1, 2 up spins
        self.basis_extended[:n, :] = self.basis_W  # 1 up spin

        for (i, idx) in enumerate(itertools.combinations(np.arange(self.n), 2)):
            self.basis_extended[n + 1 + i, list(idx)] = 1

    def fidelity(self, rbm):
        psi0 = torch.ones(self.n, device=device) / np.sqrt(self.n)
        ps = torch.exp(-rbm(self.basis_extended))
        Z = torch.sum(ps)
        psi1 = torch.sqrt(ps[:self.n] / Z)
        return (psi0 @ psi1) ** 2

    def measure_Z(self, n_measure):
        samples = self.sample(n_measure)
        samples, sample_weight = self.sample_weighting(samples)
        self.data = samples
        self.data_weight = sample_weight

    # def fidelity(self, psi):
    #     try:
    #         psi0 = self.psi
    #     except AttributeError:
    #         psi0 = self.full_psi()
    #     return (psi0.conj() @ psi)**2


class ShiftingBars(Dataset):
    def __init__(self, n, sample_size):
        super(ShiftingBars, self).__init__(n, sample_size)
        bar_length = int(n / 2)
        self.data = torch.zeros(n, n, device=device)
        for i in range(n):
            if i + bar_length <= n:
                self.data[i, i:i + bar_length] = 1
            else:
                self.data[i, i:] = 1
                self.data[i, :i + bar_length - n] = 1
        self.data_weight = torch.ones(n, device=device) / n

    def full_prob(self):
        self.idx = self.bin2dec(self.data, self.n).to(torch.int64)
        self.prob = torch.zeros(2 ** self.n, device=device)
        self.prob[self.idx] = self.data_weight
        return self.prob

    def KL_div(self, rbm):
        try:
            idx = self.idx
        except AttributeError:
            self.idx = self.bin2dec(self.data, self.n).to(torch.int64)
            idx = self.idx
        p_model = rbm.calc_prob()
        KL = torch.sum(self.data_weight * (torch.log(self.data_weight) - torch.log(p_model[idx])))
        return KL


class Ising2D_dataset(Dataset):
    def __init__(self, nx, ny, sample_size, n_measure=-1):
        n = nx * ny
        super().__init__(n, sample_size)
        self.nx = nx
        self.ny = ny
        self.Hamiltonian = Ising2D(nx, ny, 1, 1)
        if n <= 20:
            self.E_ground = self.Hamiltonian.calc_E_ground()
            self.basis = dec2bin(torch.arange(2 ** n), n)
            self.psi = torch.tensor(self.Hamiltonian.psi_ground).abs()
            self.data = self.basis
            self.data_weight = self.psi ** 2
            if n_measure > 0:
                samples = self.sample(n_measure)
                samples, sample_weight = self.sample_weighting(samples)
                self.data = samples
                self.data_weight = sample_weight
        else:
            raise NotImplementedError

    def fidelity(self, rbm):
        psi0 = self.psi
        ps = torch.exp(-rbm(self.basis))
        Z = torch.sum(ps)
        psi1 = torch.sqrt(ps / Z)
        return (psi0 @ psi1) ** 2


class ToricCode(Dataset):
    def __init__(self, nx, ny, sample_size, n_measure=-1):
        n = 2 * nx * ny
        super().__init__(n, sample_size)
        self.nx = nx
        self.ny = ny
        self.basis = dec2bin(torch.arange(2 ** n), n)
        p_idx = self.plaquette_idx()
        check = self.basis[torch.arange(2 ** n).reshape(2 ** n, 1, 1), p_idx]
        self.prob = (1 - check.sum(dim=2) % 2).prod(dim=1)
        self.prob = self.prob / self.prob.sum()
        self.psi = self.prob.sqrt()

        self.data = self.basis
        self.data_weight = self.prob

        if n_measure > 0:
            samples = self.sample(n_measure)
            samples, sample_weight = self.sample_weighting(samples)
            self.data = samples
            self.data_weight = sample_weight

    def index(self, x, y, edge):
        return 2 * ((x % self.nx) * self.ny + (y % self.ny)) + edge

    def plaquette_idx(self):
        self.p_idx = torch.zeros(self.nx * self.ny, 4, dtype=torch.int64)
        for i in range(self.nx):
            for j in range(self.ny):
                idx = i * self.ny + j
                self.p_idx[idx, 0] = self.index(i, j, 0)
                self.p_idx[idx, 1] = self.index(i, j, 1)
                self.p_idx[idx, 2] = self.index(i + 1, j, 0)
                self.p_idx[idx, 3] = self.index(i, j + 1, 1)
        return self.p_idx

    def fidelity(self, rbm):
        psi0 = self.psi
        ps = torch.exp(-rbm(self.basis))
        Z = torch.sum(ps)
        psi1 = torch.sqrt(ps / Z)
        return (psi0 @ psi1) ** 2


if __name__ == '__main__':
    dataset = Ising2D_dataset(3, 3, 10000)
    psi = dataset.psi
    import matplotlib.pyplot as plt

    plt.bar(np.arange(2 ** 9), psi, align='edge')
    plt.xlabel('Configuration')
    plt.ylabel('$|\psi|$')
