#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128GB
#SBATCH --time=5-00:30:00
#SBATCH --job-name=thomasyamada_fourier

module purge

export NUM_JULIA_THREADS=`nproc`

echo $1 > filename.txt
rundir=$SCRATCH/thomasyamada_fourier/$SLURM_JOB_ID
mkdir -p $rundir
cp fourier-driver.sbatch TY_k_omega.jl TYUtils.jl README.txt $rundir
cd $rundir

julia -t $SLURM_CPUS_PER_TASK TY_k_omega.jl > run.log

exit
