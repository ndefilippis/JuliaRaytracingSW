#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --job-name=2Lsim

module purge
module load cuda/11.6.2

export NUM_JULIA_THREADS=`nproc`

rundir=$SHAFER/twolayer_simulation/$SLURM_JOB_ID
mkdir -p $rundir
cp TwoLayerSimulation.jl Driver.jl driver.sbatch $rundir
cp Parameters.jl $rundir/Parameters.jl
cd $rundir

julia -t $SLURM_CPUS_PER_TASK Driver.jl > run.log

exit
