#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24GB
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --job-name=ray_qgsw

module purge

export NUM_JULIA_THREADS=`nproc`

rundir=$SCRATCH/swqg_rays/$SLURM_JOB_ID
mkdir -p $rundir

cp raytracing-driver.sbatch SWQG.jl RaytracingDriver.jl RaytracingMain.jl RaytracingParameters.jl README.txt $rundir
cp ../utils/SequencedOutputs.jl ../utils/IFMAB3.jl $rundir
cp ../raytracing/GPURaytracing.jl $rundir
cd $rundir

julia -t $SLURM_CPUS_PER_TASK RaytracingMain.jl > run.log

exit
