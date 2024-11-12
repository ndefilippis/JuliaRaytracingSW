#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32GB
#SBATCH --time=03:00:00
#SBATCH --job-name=plots

module purge

export NUM_JULIA_THREADS=`nproc`

rundir=$SCRATCH/swqg/plots/${SLURM_JOB_ID}

mkdir -p $rundir
cp MakeRaytracingPlots.jl RayPlotDriver.jl load_file.jl $rundir
cd $rundir
mkdir images

julia -t $SLURM_CPUS_PER_TASK RayPlotDriver.jl

exit

