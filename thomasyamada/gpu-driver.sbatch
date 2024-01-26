#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=48GB
#SBATCH --time=48:00:00
#SBATCH --gres=gpu
#SBATCH --job-name=gpu_thomasyamada_simulation

module purge

export NUM_JULIA_THREADS=`nproc`

rundir=$SCRATCH/thomasyamada_simulation/$SLURM_JOB_ID
mkdir -p $rundir
cp gpu-driver.sbatch Parameters.jl ThomasYamada.jl TYdriver.jl $rundir
cd $rundir

julia -t $SLURM_CPUS_PER_TASK TYdriver.jl GPU > run.log

exit
