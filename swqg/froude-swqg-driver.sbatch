#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15GB
#SBATCH --time=23:59:59
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --job-name=swqg_ray
#SBATCH --array=1-2
#SBATCH --output=./slurm-output/slurm-%A_%a.out

module purge

export NUM_JULIA_THREADS=`nproc`

config=froude-parameters.txt

ag=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $config)
k0=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $3}' $config)
Cg=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $4}' $config)
T=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $5}' $config)

rundir=$SCRATCH/swqg/${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}

mkdir -p $rundir
cp froude-swqg-driver.sbatch froude-parameters.txt SWQG.jl RaytracingDriver.jl RaytracingMain.jl README.txt $rundir
cp ../utils/SequencedOutputs.jl ../utils/IFMAB3.jl $rundir
cp ../raytracing/GPURaytracing.jl $rundir
cp FroudeParameters.jl $rundir/RaytracingParameters.jl
cd $rundir

julia -t $SLURM_CPUS_PER_TASK RaytracingMain.jl $ag $k0 $Cg $T > run.log

exit
