#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64GB
#SBATCH --time=01:30:00
#SBATCH --account=torch_pr_540_general
#SBATCH --job-name=2Lqg_f
#SBATCH --array=1-64

module purge

export NUM_JULIA_THREADS=`nproc`

source config-twoL.sh

rundir=$SCRATCH/2LQG_fourier/${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}
max_file=$(($(find $analysis_directory -name 2Lqg.[0-9]*.jld2 | wc -l)-1))
job_size=$(($max_K/$SLURM_ARRAY_TASK_COUNT))

mkdir -p $rundir
cp config-twoL.sh fourier-twoL.sbatch Fourier2LQG.jl README.txt $rundir
cd $rundir

julia -t $SLURM_CPUS_PER_TASK Fourier2LQG.jl $analysis_directory $max_file $SLURM_ARRAY_TASK_ID $job_size > run.log

exit
