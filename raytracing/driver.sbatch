#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=10GB
#SBATCH --time=06:00:00
#SBATCH --job-name=twolayer_raytracing

module purge
module load julia/1.6.1

export NUM_JULIA_THREADS=`nproc`

rundir=$SCRATCH/twolayer_raytracing/$SLURM_JOB_ID
mkdir -p $rundir
cp RaytracingDriver.jl TwoLayerRaytracing.jl Raytracing.jl Parameters.jl $rundir
cd $rundir

cat Parameters.jl > run.log
~/julia/my-julia -t $SLURM_CPUS_PER_TASK RaytracingDriver.jl > run.log

exit
