from mpi4py import MPI
import numpy as np
import fibomod as fb

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

numDataPerRank = 1
data = None

if rank==0:
	data = np.array([10.0,11.0,12.0,13.0])

recvbuf_nodes = np.empty(numDataPerRank,dtype='d')
comm.Scatter(data,recvbuf_nodes, root=0)
recvbuf_nodes = np.array(float(fb.calc_fibo(int(recvbuf_nodes)))) # for each input, compute the output


recvbuf_root = np.empty(4,dtype='d')
comm.Gather(recvbuf_nodes,recvbuf_root, root=0) 	

if rank == 0:
	print('Rank: ', rank, ', recvbuf received: ',recvbuf_root)
	

# print('Rank: ', rank, ',recvbuf received: ',recvbuf, ' Fibonacci number: ',fb.calc_fibo(int(recvbuf)))
