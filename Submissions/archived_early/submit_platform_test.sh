#!/bin/bash

#PBS -N baseline_platform_test
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=64GB
#PBS -l walltime=01:00:00
#PBS -q small_gpuq
#PBS -m abe
#PBS -M adam.cameron@student.uts.edu.au

export CUDA_VISIBLE_DEVICES=$(/usr/local/bin/get_idle_gpu.sh)

# Create Scratch Directory
SCRATCH="/scratch/${USER}_${PBS_JOBID%.*}"
mkdir ${SCRATCH}

cd ${PBS_O_WORKDIR}

module load cuda-latest

# Copy Project to Scratch for Fast I/O
cp -R Algorithms ${SCRATCH}/
cp -R Environments ${SCRATCH}/
mkdir ${SCRATCH}/Results

# Activate Conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate drl_nav

# Run From Scratch
cd ${SCRATCH}/Algorithms

python RunPolicy.py -r 1 -c "sac_grant_movingplat_test.yaml" -e "DRLNavigation" -t 20

# Copy Results Back
echo "Moving Results Back"
cp -R ${SCRATCH}/Results/* ${PBS_O_WORKDIR}/Results/

# Cleanup Scratch
cd ${PBS_O_WORKDIR}
rm -r ${SCRATCH}

conda deactivate
module unload cuda-latest