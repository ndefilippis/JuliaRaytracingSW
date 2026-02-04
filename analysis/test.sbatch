#!/bin/bash

#SBATCH --nodes=1
#SBATC -ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24GB
#SBATCH --time=2-12:00:00
#SBATCH --account=torch_pr_540_general
#SBATCH --job-name=qgsw

module purge

export NUM_JULIA_THREADS=`nproc`

rundir=$SCRATCH/test/$SLURM_JOB_ID
mkdir -p $rundir
cp test.sbatch Test.jl $rundir
cd $rundir

julia -t $SLURM_CPUS_PER_TASK Test.jl > run.log

exit
