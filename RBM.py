# -*- coding: utf-8 -*-
"""
Created on Sat May  1 15:19:36 2021

@author: Yuanhang Zhang
"""

import numpy as np
import itertools
import torch
import torch.nn as nn

log2 = np.log(2)
# device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RBM(nn.Module):
    def __init__(self, n, m):
        super(RBM, self).__init__()
        self.n = n
        self.m = m
        self.theta = None
        
    def init_RBM(self):
        pass
    
    def forward(self, v):
        pass

    def logexpp1(self, x):
        y = x.clone()
        mask_p = y>20
        mask_m = y<-20
        mask_0 = (~mask_p) & (~mask_m)
        y[mask_m] = 0
        y[mask_0] = torch.log1p(torch.exp(x[mask_0]))
        return y
    
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
    
    def basis_full(self):
        self.basis = self.dec2bin(torch.arange(2**self.n, device=device), self.n)
        return self.basis
        
    def basis_Jz(self, n_up):
        '''
            calculates the basis of the subspace where the number of up spins 
            equals to n_up
        '''
        n_basis = int(np.math.factorial(self.n)\
            / np.math.factorial(self.n-n_up) / np.math.factorial(n_up))
        v = torch.zeros((n_basis, self.n), device=device)
        for (i, idx) in enumerate(itertools.combinations(np.arange(self.n), n_up)):
            v[i, list(idx)] = 1
        self.basis = v
        return self.basis
    
    def calc_prob(self):
        try:
            basis = self.basis
        except AttributeError:
            basis = self.basis_full()
        E = self.forward(basis)
        self.prob = torch.exp(-E)
        self.prob /= torch.sum(self.prob)
        return self.prob

    def D_cov(self, Di):
        batch = Di[0].shape[0]
        # (batch, n_param)
        D_flattened = torch.cat([D.reshape(batch, -1) for D in Di], dim=1)
        D_mean = D_flattened.mean(dim=0)
        D_cov = torch.tensordot(D_flattened, D_flattened.conj(), dims=([0], [0])) / D_flattened.shape[0] \
                - torch.outer(D_mean, D_mean.conj())
        return D_cov

class RBM_real(RBM):
    def __init__(self, n, m):
        super(RBM_real, self).__init__(n, m)
        self.init_RBM()
    
    def init_RBM(self, scale = 0.01):
        self.a = nn.Parameter(scale * torch.randn(self.n, device=device))
        self.b = nn.Parameter(scale * torch.randn(self.m, device=device))
        self.W = nn.Parameter(scale * torch.randn(self.n, self.m, device=device))
    
    def forward(self, v):
        return -torch.matmul(v, self.a) \
            - torch.sum(self.logexpp1(self.b+torch.matmul(v, self.W)), dim=-1)
    
    def v2h(self, v, beta=1):
        theta = beta * (self.b + v @ self.W)
        h_prob = torch.sigmoid(theta)
        h = (h_prob > torch.rand(theta.shape, device=device)).to(v.dtype)
        return h, h_prob
    
    def h2v(self, h, beta=1):
        gamma = beta * (self.a + h @ self.W.T)
        v_prob = torch.sigmoid(gamma)
        v = (v_prob > torch.rand(gamma.shape, device=device)).to(h.dtype)
        return v