#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64GB
#SBATCH --time=1-12:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=1-10
#SBATCH --job-name=2Lqg_ray
#SBATCH --account=torch_pr_540_general
#SBATCH --output=./slurm-output/slurm-%A_%a.out

module purge

export NUM_JULIA_THREADS=`nproc`

config=froude-parameters.txt

config=froude-parameters.txt

ag=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $config)
k0=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $3}' $config)
Cg=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $4}' $config)
T=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $5}' $config)

rundir=$SCRATCH/twolayer_rays/${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}/

mkdir -p $rundir
cp twolayer-froude-driver.sbatch froude-parameters.txt TwoLayerQG.jl TwoLayerRaytracingMain.jl TwoLayerRaytracingDriver.jl README.txt $rundir
cp ../utils/SequencedOutputs.jl ../utils/IFMAB3.jl $rundir
cp ../raytracing/GPURaytracing.jl $rundir
cp TwoLayerFroudeParameters.jl $rundir/RaytracingParameters.jl
cd $rundir

julia -t $SLURM_CPUS_PER_TASK TwoLayerRaytracingMain.jl $ag $k0 $Cg $T > run.log

exit
