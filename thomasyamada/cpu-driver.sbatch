#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64GB
#SBATCH --time=05-05:59:00
#SBATCH --job-name=thomasyamada_simulation

module purge

export NUM_JULIA_THREADS=`nproc`

rundir=$VAST/thomasyamada_simulation/$SLURM_JOB_ID
mkdir -p $rundir
cp cpu-driver.sbatch ThomasYamada.jl TYdriver.jl TYUtils.jl cpu-setup/Parameters.jl README.txt $rundir
cd $rundir

julia -t $SLURM_CPUS_PER_TASK TYdriver.jl CPU > run.log

exit
