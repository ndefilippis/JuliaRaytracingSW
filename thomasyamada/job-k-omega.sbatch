#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=6GB
#SBATCH --time=04:30:00
#SBATCH --job-name=thomasyamada_fourier_job
#SBATCH --array=1-256

module purge

export NUM_JULIA_THREADS=`nproc`

echo $1 > filename.txt
rundir=$SCRATCH/thomasyamada_fourier/${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}
mkdir -p $rundir
cp job-k-omega.sbatch TY_k_omega.jl TYUtils.jl README.txt $rundir
cd $rundir

julia -t $SLURM_CPUS_PER_TASK TY_k_omega.jl $SLURM_ARRAY_TASK_ID > run.log

exit
