#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24GB
#SBATCH --time=1-23:00:00
#SBATCH --job-name=rsw
#SBATCH --array=1-1
#SBATCH --output=./slurm-output/slurm-%A_%a.out

module purge

export NUM_JULIA_THREADS=`nproc`

config=froude-parameters.txt

ag=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $config)
aw=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $3}' $config)
Cg=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $4}' $config)

rundir=$VAST/rsw/${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}

mkdir -p $rundir
cp froude-parameters.txt froude-rsw-driver.sbatch RotatingShallowWater.jl RSWDriver.jl RSWMain.jl README.txt $rundir 
cp ../utils/SequencedOutputs.jl ../utils/IFMAB3.jl $rundir
cp FroudeParameters.jl $rundir/Parameters.jl
cd $rundir

julia -t $SLURM_CPUS_PER_TASK RSWMain.jl $ag $aw $Cg > run.log

exit
