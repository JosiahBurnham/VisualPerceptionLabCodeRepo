#!/bin/bash
 
#SBATCH -N 1   <--- one node
#SBATCH -n 1    <---- one code per node
#SBATCH  -o log.out
#SBATCH --account=cdimattina_r_09_20   
#SBATCH --partition=silver



mpirun -n 4 python3 fibompi.py 20 21 22 23
mpirun -n 4 python3 fibompi.py 24 25 26 27
mpirun -n 4 python3 fibompi.py 28 29 30 31
mpirun -n 4 python3 fibompi.py 32 33 34 35

 
	

