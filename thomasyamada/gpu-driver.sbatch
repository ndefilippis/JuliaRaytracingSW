#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=12GB
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu
#SBATCH --job-name=thomasyamada_simulation

module purge

export NUM_JULIA_THREADS=`nproc`

rundir=$VAST/thomasyamada_simulation/$SLURM_JOB_ID
mkdir -p $rundir
cp gpu-driver.sbatch gpu-setup/Parameters.jl ThomasYamada.jl TYdriver.jl TYUtils.jl $rundir
cd $rundir

julia -t $SLURM_CPUS_PER_TASK TYdriver.jl GPU > run.log

exit
