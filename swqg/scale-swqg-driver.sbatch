#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24GB
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --job-name=swqg_ray
#SBATCH --array=1-21
#SBATCH --output=./slurm-output/slurm-%A_%a.out

module purge

export NUM_JULIA_THREADS=`nproc`

config=scale-parameters.txt

Kg1=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $config)
Kg2=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $3}' $config)
ag=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $4}' $config)
k0=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $5}' $config)

rundir=$SCRATCH/swqg/${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}

mkdir -p $rundir
mkdir -p $rundir
cp scale-swqg-driver.sbatch SWQG.jl RaytracingDriver.jl RaytracingMain.jl README.txt $rundir
cp ../utils/SequencedOutputs.jl ../utils/IFMAB3.jl $rundir
cp ../raytracing/GPURaytracing.jl $rundir
cp ScaleParameters.jl $rundir/RaytracingParameters.jl
cd $rundir

julia -t $SLURM_CPUS_PER_TASK RaytracingMain.jl $Kg1 $Kg2 $ag $k0 > run.log

exit
