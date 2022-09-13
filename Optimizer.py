# -*- coding: utf-8 -*-
"""
Created on Sun May  2 23:29:57 2021

@author: Yuanhang Zhang
"""

import numpy as np
import torch

# device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Optimizer():
    def __init__(self, rbm, dataset, sampler, mode_sampler=None, lr=0.1):
        self.rbm = rbm
        self.dataset = dataset
        self.sampler = sampler
        self.mode_sampler = mode_sampler
        self.cd_iters = 1
        self.lr = lr
        self.lr_max = 1
        self.lr_min = 0.001
        self.optimizer = torch.optim.Adam(rbm.parameters(), lr=self.lr)
        self.print_interval = 500
        
        # self.v_pcd = torch.randint(0, 2, [100, self.rbm.n]).to(torch.get_default_dtype())

    def calc_gradient(self, lr):
        batch = self.dataset.sample_size
        data = self.dataset.sample()
        [h, _] = self.rbm.v2h(data)
        vh_data = data.T @ h / batch
        v_data = torch.sum(data, 0)/batch
        h_data = torch.sum(h, 0)/batch
        D_data = [v_data, h_data, vh_data]
        
        v = self.sampler(self.rbm, data)
        [h, _] = self.rbm.v2h(v)
        vh_model = v.T @ h / batch
        v_model = torch.sum(v, 0) / batch
        h_model = torch.sum(h, 0) / batch
        D_model = [v_model, h_model, vh_model]

        for i, param in enumerate(self.rbm.parameters()):
            Gi = D_model[i] - D_data[i]
            param.grad = lr * Gi

    def calc_gradient_mode(self, lr):
        batch = self.dataset.sample_size
        data = self.dataset.sample()
        [h, _] = self.rbm.v2h(data)
        vh_data = data.T @ h / batch
        v_data = torch.sum(data, 0)/batch
        h_data = torch.sum(h, 0)/batch
        D_data = [v_data, h_data, vh_data]
        
        [v, mode_push] = self.mode_sampler(self.rbm)
        [h, h_prob] = self.rbm.v2h(v)
        
        vh_model = torch.outer(v, h_prob)
        v_model = v
        h_model = h_prob
        D_model = [v_model, h_model, vh_model]

        for i, param in enumerate(self.rbm.parameters()):
            Gi = D_model[i] - D_data[i]
            param.grad = mode_push * lr * Gi
    
    @torch.no_grad()
    def train(self, n_epoch, max_mode_prob=0, calc_f=False, id=0):
        # lr = np.linspace(self.lr_max, self.lr_min, n_epoch)
        sigmoid_midpoint = n_epoch*3/10
        sigmoid_width = n_epoch/20
        if calc_f:
            f_store = np.zeros(int(np.ceil(n_epoch/self.print_interval)))
        else:
            f_store = None
        lr = self.lr
        f_best = 0
        epochs_without_improvement = 0
        for i in range(n_epoch):
            # lr = self.lr_min + (self.lr_max-self.lr_min) * ((i-n_epoch)/n_epoch)**2
            # lr = self.lr
            self.optimizer.zero_grad()
            mode_prob = max_mode_prob / (1+np.exp(-(i-sigmoid_midpoint)/sigmoid_width))
            if mode_prob > np.random.rand():
                self.calc_gradient_mode(1)
            else:
                self.calc_gradient(1)
            self.optimizer.step()
            if i % self.print_interval == 0:
                if calc_f:
                    f = self.dataset.fidelity(self.rbm)
                    f_store[i//self.print_interval] = f.detach().cpu().numpy()
                    print(f'id={id}  i={i}  n={self.rbm.n}  f={f:.6f}  lr={lr:.4e}  mode_prob={max_mode_prob:.2f}  n_measure={self.dataset.n_measure}')   
                    if f > f_best:
                        f_best = f
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1
                        if epochs_without_improvement > 20:
                            f_best = f
                            lr /= 2
                            epochs_without_improvement = 0
                            for g in self.optimizer.param_groups:
                                g['lr'] = lr
                else:
                    print('i = {}'.format(i))    
            # if i % 5000 == 0:
            #     torch.save(self.rbm.state_dict(), 'rbm_{}.ckpt'.format(i))
        return f_store
