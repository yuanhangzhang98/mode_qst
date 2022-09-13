# -*- coding: utf-8 -*-
"""
Created on Sun May  2 22:29:15 2021

@author: Yuanhang Zhang
"""

import numpy as np
import torch

# device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Sampler:
    def __init__(self):
        pass
    
    def sample_weighting(self, samples):
        samples, sample_weight = torch.unique(samples, dim=0, return_counts=True)
        sample_weight = sample_weight / torch.sum(sample_weight)
        return samples, sample_weight

class ExactSampler(Sampler):
    def __init__(self):
        return
    
    def __call__(self, rbm):
        try:
            basis = rbm.basis
        except AttributeError:
            if rbm.U1_symm:
                basis = rbm.basis_Jz(int(rbm.n/2))
            else:
                basis = rbm.basis_full()
        E = rbm(basis)
        try:
            E = 2*E.real
        except RuntimeError:
            E = 2*E
        p = torch.exp(-E)
        p = p/torch.sum(p)
        return rbm.basis, p

class CDSampler(Sampler):
    # contrastive divergence
    def __init__(self, cd_iters):
        self.cd_iters = cd_iters
        
    def __call__(self, rbm, v):
        for i in range(self.cd_iters):
            [h, _] = rbm.v2h(v)
            v = rbm.h2v(h)
        return v

class PCDSampler(Sampler):
    # contrastive divergence
    def __init__(self, cd_iters):
        self.cd_iters = cd_iters
        self.v_pcd = None
        
    def __call__(self, rbm, v):
        if self.v_pcd is None:
            self.v_pcd = v
        v = self.v_pcd
        for i in range(self.cd_iters):
            [h, _] = rbm.v2h(v)
            v = rbm.h2v(h)
        self.v_pcd = v
        return v

class PTSampler(Sampler):
    # parallel tempering
    def __init__(self, cd_iters):
        self.cd_iters = cd_iters
        self.v_last = None
        self.n_T = None
        
    def __call__(self, rbm, v):
        if self.n_T == None:
            self.n_T = int(np.round(np.sqrt(rbm.n)))
            if self.n_T < 2:
                self.n_T = 2
            self.log_T_ratio = np.log(100.0 ** (1.0/(self.n_T-1)))
            self.beta0 = torch.exp(-self.log_T_ratio *\
                    torch.arange(self.n_T).to(torch.get_default_dtype()))
            self.dbeta = torch.diff(self.beta0, dim=0) # (n_T-1)
            self.beta0 = self.beta0.reshape(-1, 1, 1) # (n_T, 1, 1)
            self.batch, n = v.shape
            self.v_last = torch.randint(0, 2, [self.n_T, self.batch, rbm.n]).to(torch.get_default_dtype())

        beta = self.beta0

        v = self.v_last # (n_T, batch, n)
        for i in range(self.cd_iters):
            [h, _] = rbm.v2h(v, beta)
            v = rbm.h2v(h, beta)
        E = rbm.forward(v) # (n_T, batch)
        
        for j in range(self.n_T-1):
            swap_mask = (((self.dbeta[j] * (E[j+1]-E[j])).exp() > torch.rand(self.batch))).unsqueeze(1) # (batch, 1)
            t0 = v[j+1] * swap_mask + v[j] * ~swap_mask
            t1 = v[j] * swap_mask + v[j+1] * ~swap_mask
            v[[j, j+1]] = torch.stack([t0, t1], dim=0)
            
        self.v_last = v
        return v[0]

class ExactModeSampler(Sampler):
    def __init__(self):
        return
    
    def __call__(self, rbm):
        try:
            basis = rbm.basis
        except AttributeError:
            basis = rbm.basis_full()
        E = rbm(basis)
        try:
            E = E.real
        except RuntimeError:
            pass
        [E, idx] = torch.min(E, dim=0)
        v = basis[idx, :]
        theta = rbm.b + torch.matmul(v, rbm.W)
        h = (torch.sigmoid(theta) > torch.rand(theta.shape, device=device)).to(torch.get_default_dtype())
        E_mode = -(v @ rbm.W @ h + v @ rbm.a + h @ rbm.b)\
                + torch.sum(rbm.a)/2 + torch.sum(rbm.b)/2 + torch.sum(rbm.W)/4
        mode_push = -E_mode / (rbm.n+1) / (rbm.m+1) / 4
        return v, mode_push.clip(0, 1)

class ExactJointModeSampler(Sampler):
    def __init__(self):
        return
    
    def __call__(self, rbm):
        basis = rbm.dec2bin(torch.arange(2**(rbm.n+rbm.m), device=device), rbm.n+rbm.m)
        v = basis[:, :rbm.n]
        h = basis[:, rbm.n:]
        E = -(v @ rbm.a + h @ rbm.b + torch.einsum('bn, bm, nm->b', v, h, rbm.W))
        try:
            E = E.real
        except RuntimeError:
            pass
        [E, idx] = torch.min(E, dim=0)
        v = basis[idx, :rbm.n]
        h = basis[idx, rbm.n:]
        E_mode = E + torch.sum(rbm.a)/2 + torch.sum(rbm.b)/2 + torch.sum(rbm.W)/4
        mode_push = -E_mode / (rbm.n+1) / (rbm.m+1) / 4
        return v, h, mode_push.clip(0, 1)

class PTModeSampler(Sampler):
    def __init__(self, n_sample):
        self.pt = PTSampler(n_sample)
        
    def __call__(self, rbm, n_mode=1):
        samples, _ = self.pt(rbm)
        E = - torch.matmul(samples, rbm.a) \
            - torch.sum(rbm.logcosh(rbm.b+torch.matmul(samples, rbm.W)), dim=-1)
        [E, idx] = torch.topk(E, n_mode, largest=False)
        v = samples[idx, :]
        theta = rbm.b + torch.matmul(v, rbm.W)
        h = (2*(torch.sigmoid(2*theta) > \
            torch.rand(theta.shape, device=device))-1).to(torch.get_default_dtype())
        if n_mode == 1:
            sample_weight = torch.tensor([1.0], device=device)
        else:
            p = torch.exp(-E)
            sample_weight = p / torch.sum(p)
        E_mode = -(v[0, :] @ rbm.W @ h[0, :] + v[0, :] @ rbm.a + h[0, :] @ rbm.b)
        mode_push = -E_mode / (rbm.n+1) / (rbm.m+1)
        return v, sample_weight, mode_push.clip(0, 1)
    
class WModeSampler(Sampler):
    def __init__(self):
        return
    
    def __call__(self, rbm):
        basis = torch.eye(rbm.n)
        E = rbm(basis)
        [E, idx] = torch.min(E, dim=0)
        v = basis[idx, :]
        theta = rbm.b + torch.matmul(v, rbm.W)
        h = (torch.sigmoid(theta) > torch.rand(theta.shape, device=device)).to(torch.get_default_dtype())
        E_mode = -(v @ rbm.W @ h + v @ rbm.a + h @ rbm.b)\
                + torch.sum(rbm.a)/2 + torch.sum(rbm.b)/2 + torch.sum(rbm.W)/4
        mode_push = -E_mode / (rbm.n+1) / (rbm.m+1) / 4
        return v, mode_push.clip(0, 1)

class MemModeSampler(Sampler):
    def __init__(self):
        # [alpha, beta, gamma, delta, epsilon, zeta, dt]
        self.param = torch.Tensor([5, 20, 0.25, 0.05, 0.001, 0.1, 1]).to(device)
        self.max_step = 1000
        self.dt_min = 2 ** (-7)
        self.dt_max = 1
        self.n_restart = 10
        self.dt_last = self.dt_min * torch.ones(self.n_restart, 1, device=device)
        self.noise = 1e-2
    
    @torch.no_grad()
    def __call__(self, rbm):
        self.rbm2maxsat(rbm)
        factor = torch.zeros(7, device=device)
        max_xl = 10 * self.n * self.m
        E_min = np.inf
        
        v = 2 * torch.rand(self.n_restart, self.n, device=device) - 1
        h = 2 * torch.rand(self.n_restart, self.m, device=device) - 1
        xl = (1 + self.W2).repeat(self.n_restart, 1, 1)
        v1 = torch.cat([v, torch.ones(self.n_restart, 1, device=device)], dim=1).reshape(self.n_restart, self.n+1, 1)
        h1 = torch.cat([h, torch.ones(self.n_restart, 1, device=device)], dim=1).reshape(self.n_restart, 1, self.m+1)
        Cv = (1 - self.qv*v1) / 2
        Ch = (1 - self.qh*h1) / 2
        xs = torch.min(Cv, Ch)
        for i in range(self.max_step):                    
            dv, dh, dxs, dxl, C = self.timestep(factor, v, h, xl, xs)
            n_unsat = torch.sum((C>=0.5) & (self.W2>0), dim=[1, 2])
            v = torch.clip(v+dv, -1, 1)
            h = torch.clip(h+dh, -1, 1)
            xs = torch.clip(xs+dxs, 0, 1)
            xl = torch.clip(xl+dxl, 1, max_xl)
            v1 = ((v + self.noise * torch.randn(v.shape, device=device))>0).to(torch.get_default_dtype())
            h1 = ((h + self.noise * torch.randn(h.shape, device=device))>0).to(torch.get_default_dtype())
            E = -(v1 @ self.a + h1 @ self.b + torch.einsum('bn, nm, bm->b', v1, self.W, h1))
            [E, idx] = torch.min(E, dim=0)
            if E<E_min:
                E_min = E
                v_mode = v1[idx, :]
                h_mode = h1[idx, :]
            # print(k, i, (E+self.const).detach().cpu().numpy(), n_unsat.detach().cpu().numpy())                
                
        E_mode = E_min + self.const
        mode_push = -E_mode / (self.n+1) / (self.m+1) / 4
        return v_mode, mode_push.clip(0, 1)
        
    def rbm2maxsat(self, rbm):
        '''
            QUBO to Max-2-SAT
            Qij > 0  ===>  Qij (~yi) | (~yj)
            Qij < 0  ===> |Qij|( yi  | (~yj))   &   |Qij| yj
            
            1-SAT:
            bi > 0  ===>  bi (~yi)
            bi < 0  ===>  bi yi
        '''
        self.n = rbm.n
        self.m = rbm.m
        # self.a = 2 * rbm.a - 2 * (rbm.W @ torch.ones(rbm.m, 1, device=device));
        # self.b = 2 * rbm.b - 2 * (torch.ones(1, rbm.n, device=device) @ rbm.W);
        # self.W = 4 * rbm.W
        # self.const = torch.sum(rbm.a) + torch.sum(rbm.b) - torch.sum(rbm.W);
        
        self.a = rbm.a
        self.b = rbm.b
        self.W = rbm.W
        self.const = torch.sum(rbm.a)/2 + torch.sum(rbm.b)/2 + torch.sum(rbm.W)/4
        
        self.Q = torch.zeros(rbm.n+rbm.m, rbm.n+rbm.m, device=device)
        self.Q[:rbm.n, :rbm.n] = -torch.diag(self.a)
        self.Q[rbm.n:, rbm.n:] = -torch.diag(self.b)
        self.Q[:rbm.n, rbm.n:] = -self.W
        self.Q /= torch.linalg.norm(self.Q)
        a1 = torch.diag(self.Q)[:rbm.n]
        b1 = torch.diag(self.Q)[rbm.n:]
        b1 += torch.sum(self.Q[:rbm.n, rbm.n:]*(self.Q[:rbm.n, rbm.n:]<0), dim=0)
        self.W2 = torch.zeros(rbm.n+1, rbm.m+1, device=device)
        self.W2[:rbm.n, :rbm.m] = self.Q[:rbm.n, rbm.n:]
        self.W2[:rbm.n, -1] = a1
        self.W2[-1, :rbm.m] = b1
        self.E_sat_const = torch.sum(self.Q[self.Q<0])
        self.qv = -torch.sign(self.W2)
        self.qv[-1, :-1] = -1
        self.qv[-1, -1] = 1
        self.qh = -torch.ones_like(self.W2, device=device)
        self.qh[-1, :-1] = -torch.sign(b1)
        self.qh[-1, -1] = 1
        self.W2 = torch.abs(self.W2)
        
    def timestep(self, factor, v, h, xl, xs):
        [alpha, beta, gamma, delta, epsilon, zeta, dt_factor] = self.param * 2**factor
        v = torch.cat([v, torch.ones(self.n_restart, 1, device=device)], dim=1).reshape(self.n_restart, self.n+1, 1)
        h = torch.cat([h, torch.ones(self.n_restart, 1, device=device)], dim=1).reshape(self.n_restart, 1, self.m+1)
        Cv = (1 - self.qv*v) / 2
        Ch = (1 - self.qh*h) / 2
        C = torch.min(Cv, Ch)
        Gv = self.qv * Ch
        Gh = self.qh * Cv
        mask = Cv < Ch
        Rv = torch.zeros(self.n_restart, self.n+1, self.m+1, device=device)
        Rh = torch.zeros(self.n_restart, self.n+1, self.m+1, device=device)
        Rv[mask] = self.qv.repeat(self.n_restart, 1, 1)[mask] * C[mask]
        Rh[~mask] = self.qh.repeat(self.n_restart, 1, 1)[~mask] * C[~mask]
        G_factor = self.W2 * xl * xs
        R_factor = self.W2 * (1 + zeta*xl) * (1 - xs)
        dv = torch.sum((G_factor * Gv + R_factor * Rv), dim=2)[:, :-1]
        dh = torch.sum((G_factor * Gh + R_factor * Rh), dim=1)[:, :-1]
        dxs = beta * (xs + epsilon) * (C - gamma)
        dxl = alpha * (1 + self.W2) * (C - delta)
        
        v = v.reshape(self.n_restart, -1)[:, :-1]
        h = h.reshape(self.n_restart, -1)[:, :-1]
        
        v1 = torch.clip(v+dv*self.dt_last, -1, 1)
        h1 = torch.clip(h+dh*self.dt_last, -1, 1)
        dv1 = torch.abs(v1 - v)
        dh1 = torch.abs(h1 - h)
        
        [max_dv, _] = torch.max(torch.cat([dv1, dh1], dim=1), dim=1)
        dt = dt_factor / (max_dv/self.dt_last.reshape(max_dv.shape))
        dt = torch.clip(dt, self.dt_min, self.dt_max).unsqueeze(-1)
        self.dt_last = dt
        
        return dv*dt, dh*dt, dxs*dt.unsqueeze(-1), dxl*dt.unsqueeze(-1), C
        
    def weighted_sat_sum(self, v, h):
        batch = v.shape[0]
        v = torch.cat([2*v-1, torch.ones(batch, 1, device=device)], dim=1)
        h = torch.cat([2*h-1, torch.ones(batch, 1, device=device)], dim=1)
        v = v.reshape(batch, self.n+1, 1).repeat(1, 1, self.m+1)
        h = h.reshape(batch, 1, self.m+1).repeat(1, self.n+1, 1)
        Cv = (1 - self.qv*v)/2
        Ch = (1 - self.qh*h)/2
        C = torch.min(Cv, Ch)
        E = torch.sum((C>=0.5) * self.W2, dim=[1, 2])
        E += self.E_sat_const
        return E
