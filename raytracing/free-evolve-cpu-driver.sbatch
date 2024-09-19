#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64GB
#SBATCH --time=6-23:59:59
#SBATCH --job-name=twolayer_raytracing

module purge

export NUM_JULIA_THREADS=`nproc`

rundir=$SCRATCH/twolayer_raytracing/$SLURM_JOB_ID
mkdir -p $rundir
cp cpu-driver.sbatch Driver.jl TwoLayerRaytracing.jl Raytracing.jl README.txt $rundir
cp initial_conditions/initial_condition_512x512_U=1.10_freely_evolve.jld2 $rundir/initial_condition.jld2
cp CPUParameters.jl $rundir/Parameters.jl
cd $rundir

#~/julia-1.9.4/julia-writable -t $SLURM_CPUS_PER_TASK Driver.jl > run.log
julia -t $SLURM_CPUS_PER_TASK Driver.jl > run.log

exit
