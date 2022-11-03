

---- begin on next line:

#!/bin/bash

#SBATCH -N 1   <--- one node

#SBATCH -n 1    <---- one code per node

#SBATCH  -o log.out

#SBATCH --account=    <---- enter your billing account name here

#SBATCH --partition=silver

#SBATCH --time=1:00:00

 

# source modules required for the job:

#module load ab_initio/psi4/psi4

#module load ab_initio/psi4/psi4

#module load mpi/openmpi-4.0.3v/openmpi-4.0.3v (add this line for parallel).

 

# loop over all input (*.in) files and run them with the Q-Chem program package

for f in `ls *.in`; do

  qchem -nt $SLURM_NTASKS ${f} ${f/.in/.out}

done

--- EOF

 

You would replace the line with qchem with your Python call. Before using, you'll need to make the file executable (chmod 750 submit.sh). Submit via "sbatch submit.sh"

