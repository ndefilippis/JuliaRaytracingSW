#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=20GB
#SBATCH --time=12:00:00
#SBATCH --job-name=twolayer_simulation

module purge

export NUM_JULIA_THREADS=`nproc`

rundir=$SCRATCH/twolayer_simulation/$SLURM_JOB_ID
mkdir -p $rundir
cp TwoLayerSimulation.jl $rundir
cd $rundir

~/julia-1.9.4/julia -t $SLURM_CPUS_PER_TASK TwoLayerSimulation.jl > run.log

exit
