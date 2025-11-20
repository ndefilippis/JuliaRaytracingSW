#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12GB
#SBATCH --time=04:30:00
#SBATCH --job-name=rsw_f
#SBATCH --array=1-129

module purge

export NUM_JULIA_THREADS=`nproc`

source config.sh

rundir=$SCRATCH/rsw_fourier/${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}
max_file=$(($(find $analysis_directory -name rsw.[0-9]*.jld2 | wc -l)-1))
job_size=$(($max_K/$num_tasks))

mkdir -p $rundir
cp config.sh fourier-job.sbatch FourierRSW.jl README.txt $rundir
cp ../RSWUtils.jl $rundir
cd $rundir

julia -t $SLURM_CPUS_PER_TASK FourierRSW.jl $analysis_directory $max_file $SLURM_ARRAY_TASK_ID $job_size > run.log

exit
