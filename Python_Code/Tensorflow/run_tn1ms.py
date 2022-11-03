# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 16:22:19 2020

@author: cdimattina
"""

import sys
import texnet1_multiscale_parallel as tn1

num_args = len(sys.argv)
args     = sys.argv

if(num_args < 7):
    print("Usage: run_tn1ms <fname> <num_files> <num_v2> <num_v4> <l2penalty> <sparse_penalty>")
else:
    fname       = str(args[1])
    num_files   = int(args[2])
    num_v2      = int(args[3])
    num_v4      = int(args[4])
    l2penalty   = float(args[5])
    sparse_penalty = float(args[6])
    
    
    tn1.train_model(fname,num_files,num_v2,num_v4,l2penalty,sparse_penalty)
    
