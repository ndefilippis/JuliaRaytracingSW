#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24GB
#SBATCH --time=05:59:59
#SBATCH --gres=gpu:v100:1
#SBATCH --job-name=swqg_ray
#SBATCH --mail-user=nad9961@nyu.edu
#SBATCH --mail-type=BEGIN,FAIL
#SBATCH --array=1-2
#SBATCH --output=./slurm-output/slurm-%A_%a.out

module purge

export NUM_JULIA_THREADS=`nproc`

config=merger-parameters.txt

snapfile=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $config)
snapkey=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $3}' $config)

rundir=$SCRATCH/swqg/${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}

mkdir -p $rundir
cp merger-swqg-driver.sbatch merger-parameters.txt SWQG.jl RaytracingDriver.jl RaytracingMain.jl README.txt $rundir
cp ../utils/SequencedOutputs.jl ../utils/IFMAB3.jl $rundir
cp ../raytracing/GPURaytracing.jl $rundir
cp MergerParameters.jl $rundir/RaytracingParameters.jl
cd $rundir

julia -t $SLURM_CPUS_PER_TASK RaytracingMain.jl $snapfile $snapkey > run.log

exit
