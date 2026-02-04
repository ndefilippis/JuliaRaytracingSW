#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=16GB
#SBATCH --time=3:59:59
#SBATCH --gres=gpu:1
#SBATCH --job-name=test
#SBATCH --comment="gpu_mps=yes"
#SBATCH --account=torch_pr_540_general

module purge

export NUM_JULIA_THREADS=`nproc`

rundir=$SCRATCH/test/${SLURM_JOB_ID}

mkdir -p $rundir
cp Test.jl $rundir
cd $rundir

julia -t $SLURM_CPUS_PER_TASK Test.jl > run.log

exit
