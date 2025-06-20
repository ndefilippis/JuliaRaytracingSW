#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15GB
#SBATCH --time=23:59:59
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --job-name=rsw_ray
#SBATCH --output=./slurm-output/slurm-%A.out

module purge

export NUM_JULIA_THREADS=`nproc`

rundir=$SCRATCH/rsw_ray/${SLURM_JOB_ID}/

mkdir -p $rundir
cp rsw-raytracing-driver.sbatch RotatingShallowWater.jl RSWRaytracingDriver.jl RSWRaytracingMain.jl README.txt $rundir
cp ../utils/SequencedOutputs.jl ../utils/IFMAB3.jl $rundir
cp ../raytracing/GPURaytracing.jl ../raytracing/RaytracingDriver.jl $rundir
cp RSWRaytracingParameters.jl $rundir/RaytracingParameters.jl
cd $rundir

julia -t $SLURM_CPUS_PER_TASK RSWRaytracingMain.jl > run.log

exit
