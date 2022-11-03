"""
fibompi.py

Description: This computes the fibonacci sequence for multiple inputs
	     using a parallel process (4 cores only). This program expects 4 inputs. 
	     
	     Basic fibonacci sequence calculation
		base case: 		
		f(1) = 1	
		f(2) = 1
		recursion: 
		f(n) = f(n-1) + f(n-2) , n > 2 

Usage: 	     python3 fibompi.py <n1> <n2> <n3> <n4>

"""

import sys
import time
import fibomod as fb
from mpi4py import MPI
import numpy as np

num_args = len(sys.argv)
args     = sys.argv


def check_valid(inlist):
	is_valid = 1
	for i in range(1,len(inlist)):
		if(int(inlist[i])<1):
			is_valid = 0
	return is_valid

# Make sure there were 4 arguments to the program
if(num_args == 5):

	is_valid = check_valid(args) 
	# For all the inputs, sequentially compute the requested Fibonacci number
	if(is_valid):
		

		comm = MPI.COMM_WORLD
		size = comm.Get_size()
		rank = comm.Get_rank()

		numDataPerRank = 1
		data = None

		if rank==0:
			t0   = time.time()
			data = np.array(args[1:5],dtype='d')
		

		recvbuf_nodes = np.empty(numDataPerRank,dtype='d')
		comm.Scatter(data,recvbuf_nodes, root=0)
		recvbuf_nodes = np.array(float(fb.calc_fibo(int(recvbuf_nodes)))) # for each input, compute the output

		recvbuf_root = np.empty(4,dtype='d')
		comm.Gather(recvbuf_nodes,recvbuf_root, root=0) 	

		if rank==0:
			for i in range(4):
				outStr = "The " + str(int(data[i])) + "-th Fibonacci number is: " + str(int(recvbuf_root[i]))
				print(outStr)
		
			t1 = time.time()
			wt = t1-t0                
			timeStr = "Wall time = " + str("{:.4}".format(wt))  + " seconds"
			print(timeStr)
	else:
		print("Usage: fibompi <n1> <n2> <n3> <n4>, all inputs integers > 0")	
else:
	print("Usage: fibompi <n1> <n2> <n3> <n4>, all inputs integers > 0")

	

