#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24GB
#SBATCH --time=1-12:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --account=torch_pr_540_general
#SBATCH --array=1-4
#SBATCH --job-name=2Lqg
#SBATCH --output=./slurm-output/slurm-%A_%a.out

module purge

export NUM_JULIA_THREADS=`nproc`

config=fourier-parameters.txt

ag=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $config)

rundir=$SHAFERLAB/2Lqg/${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}/
mkdir -p $rundir
cp twolayer-fourier-driver.sbatch TwoLayerQG.jl TwoLayerDriver.jl TwoLayerMain.jl README.txt $rundir
cp ../utils/SequencedOutputs.jl ../utils/IFMAB3.jl $rundir
cp TwoLayerFourierParameters.jl $rundir/Parameters.jl
cd $rundir

julia -t $SLURM_CPUS_PER_TASK TwoLayerMain.jl $ag > run.log

exit
