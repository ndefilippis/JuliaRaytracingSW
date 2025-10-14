#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=16GB
#SBATCH --time=1-23:59:59
#SBATCH --gres=gpu:v100:1
#SBATCH --job-name=rsw_ray
#SBATCH --mail-user=nad9961@nyu.edu
#SBATCH --mail-type=BEGIN,FAIL,COMPLETED
#SBATCH --array=1-3
#SBATCH --output=./slurm-output/slurm-%A_%a.out

module purge

export NUM_JULIA_THREADS=`nproc`

config=rsw-ray-parameters.txt

ag=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $config)

rundir=$SCRATCH/rsw_ray/${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}

mkdir -p $rundir
cp rsw-raytracing-driver.sbatch RotatingShallowWater.jl RSWRaytracingDriver.jl RSWRaytracingMain.jl README.txt $rundir
cp ../utils/SequencedOutputs.jl ../utils/IFMAB3.jl $rundir
cp ../raytracing/GPURaytracing.jl ../raytracing/RaytracingDriver.jl $rundir
cp RSWRaytracingJobParameters.jl $rundir/RaytracingParameters.jl
cd $rundir

julia -t $SLURM_CPUS_PER_TASK RSWRaytracingMain.jl $ag > run.log

exit
