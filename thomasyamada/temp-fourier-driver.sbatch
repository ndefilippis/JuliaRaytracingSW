#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128GB
#SBATCH --time=5-00:30:00
#SBATCH --job-name=thomasyamada_fourier

module purge

export NUM_JULIA_THREADS=`nproc`

echo $1 > filename.txt
rundir=$SCRATCH/thomasyamada_fourier/$SLURM_JOB_ID
mkdir -p $rundir
cp temp-fourier-driver.sbatch temp_TYFourierAnalysis.jl TYUtils.jl README.txt $rundir
cd $rundir

julia -t $SLURM_CPUS_PER_TASK temp_TYFourierAnalysis.jl > run.log

exit
