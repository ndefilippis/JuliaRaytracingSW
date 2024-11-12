#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24GB
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=rsw

module purge

export NUM_JULIA_THREADS=`nproc`

rundir=$SCRATCH/rsw/$SLURM_JOB_ID
mkdir -p $rundir
cp rsw-driver.sbatch RotatingShallowWater.jl RSWDriver.jl RSWMain.jl README.txt $rundir
cp ../utils/SequencedOutputs.jl ../utils/IFMAB3.jl $rundir
cp RSWParameters.jl $rundir/Parameters.jl
cd $rundir

julia -t $SLURM_CPUS_PER_TASK RSWMain.jl > run.log

exit
