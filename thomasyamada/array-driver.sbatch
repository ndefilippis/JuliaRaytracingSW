#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10GB
#SBATCH --time=8:00:00
#SBATCH --gres=gpu
#SBATCH --job-name=TY_sim
#SBATCH --array=1-4

module purge

export NUM_JULIA_THREADS=`nproc`

config=parameters.txt

Eg=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $config)
Ew=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $3}' $config)

rundir=$SCRATCH/thomasyamada_simulation/${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}
mkdir -p $rundir
cp array-driver.sbatch ThomasYamada.jl TYdriver.jl TYUtils.jl $rundir
cp ArrayParameters.jl $rundir/Parameters.jl
cd $rundir

julia -t $SLURM_CPUS_PER_TASK TYdriver.jl GPU $Eg $Ew > run.log

exit
