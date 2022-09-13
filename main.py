# -*- coding: utf-8 -*-
"""
Created on Mon May  3 00:07:49 2021

@author: Yuanhang Zhang
"""

from RBM import RBM_real
from Dataset import W_state
from Sampler import WModeSampler, CDSampler, PCDSampler, PTSampler
from Optimizer import Optimizer

import sys
import os
import time
import numpy as np
import torch
import torch.multiprocessing as mp
from itertools import repeat
# import matplotlib
# import matplotlib.pyplot as plt

torch.set_default_tensor_type(torch.cuda.DoubleTensor\
                              if torch.cuda.is_available()\
                                  else torch.DoubleTensor)
    


def train(args):
    i, n, n_sample, n_measure, sampler, is_mode = args
    # torch.manual_seed(i)
    alpha = 2
    m = alpha*n
    lr = 0.01
    n_epoch = 200001
    max_mode_prob = 0.1
    print('Start Process', i)
    rbm = RBM_real(n, m)
    dataset = W_state(n, n_sample, n_measure)
    mode_sampler = WModeSampler()
    optimizer = Optimizer(rbm, dataset, sampler, mode_sampler, lr)
    if is_mode:
        f_store = optimizer.train(n_epoch, max_mode_prob=max_mode_prob, calc_f=True, id=i)
    else:
        f_store = optimizer.train(n_epoch, max_mode_prob=0, calc_f=True, id=i)
    with open('results/stat_CD_{}_{}_{}_{}.npy'.format(n, is_mode, n_measure, i), 'wb') as f:
        np.save(f, f_store)
    # torch.save(rbm.state_dict(), 'results/W_{}.ckpt'.format(n))
    return f_store

if __name__ == '__main__':
    try:
        os.mkdir('results/')
    except:
        pass
    
    n = 10
    n_sample = n**2
    n_measure = 32 * n
    sampler = CDSampler(cd_iters=1)
    # train([0, n, n_sample, n_measure, False])
    f_store = train([0, n, n_sample, n_measure, sampler, False])
    
    

    # Repeat the training for ensemble_size * n_repeat times, take the median of the results
    
    # mp.set_start_method('spawn')
    # ensemble_size = 5
    # n_repeat = 1
    # ns = np.array([10, 15, 20, 25, 30, 35, 40, 45, 50])
    # n_samples = ns**2
    # n_measure_factor = np.array([2, 8, 32])
    # pool = mp.Pool(ensemble_size)
    # sampler = CDSampler(cd_iters=1)
    #
    # for j in range(n_repeat):
    #     for i, n in enumerate(ns):
    #         n_sample = n_samples[i]
    #         n_measures = n * n_measure_factor
    #         for n_measure in n_measures:
    #             pool.map(train, \
    #                 zip(range(j*ensemble_size, (j+1)*ensemble_size), repeat(n), repeat(n_sample), repeat(n_measure), repeat(sampler), repeat(False)))
    # pool.close()
