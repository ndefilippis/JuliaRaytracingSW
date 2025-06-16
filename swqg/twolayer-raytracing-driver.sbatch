#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=16GB
#SBATCH --time=2-23:00:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --job-name=2Lqg_ray
#SBATCH --output=./slurm-output/slurm-%A.out

module purge

export NUM_JULIA_THREADS=`nproc`

rundir=$SCRATCH/twolayer_rays/${SLURM_JOB_ID}/

mkdir -p $rundir
cp twolayer-raytracing-driver.sbatch TwoLayerQG.jl TwoLayerMain.jl README.txt $rundir
cp TwoLayerRaytracingDriver.jl $rundir/RaytracingDriver.jl
cp ../utils/SequencedOutputs.jl ../utils/IFMAB3.jl $rundir
cp ../raytracing/GPURaytracing.jl $rundir
cp TwoLayerRaytracingParameters.jl $rundir/RaytracingParameters.jl
cd $rundir

julia -t $SLURM_CPUS_PER_TASK TwoLayerMain.jl > run.log

exit
