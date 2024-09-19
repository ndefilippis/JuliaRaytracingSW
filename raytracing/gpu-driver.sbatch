#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=48GB
#SBATCH --time=3-00:00:00
#SBATCH --job-name=twolayer_raytracing

module purge

export NUM_JULIA_THREADS=`nproc`

rundir=$SCRATCH/twolayer_raytracing/$SLURM_JOB_ID
mkdir -p $rundir

cp gpu-driver.sbatch Driver.jl README.txt $rundir

cp GPUTwoLayerRaytracing.jl $rundir/TwoLayerRaytracing.jl
cp GPURaytracing.jl $rundir/Raytracing.jl
cp GPUParameters.jl $rundir/Parameters.jl

cp initial_conditions/initial_condition_512x512_U=0.10.jld2 $rundir/initial_condition.jld2
cd $rundir

julia -t $SLURM_CPUS_PER_TASK Driver.jl > run.log

exit
