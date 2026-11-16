#!/bin/bash

#PBS -N baseline_hz
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=64GB
#PBS -l walltime=15:00:00
#PBS -q small_gpuq
#PBS -m abe
#PBS -M adam.cameron@student.uts.edu.au

SCRATCH="/scratch/${USER}_${PBS_JOBID%.*}"
mkdir ${SCRATCH}

cd ${PBS_O_WORKDIR}
module load cuda/11.8

cp -R Algorithms ${SCRATCH}/
cp -R Environments ${SCRATCH}/
mkdir ${SCRATCH}/Results

source ~/miniconda3/etc/profile.d/conda.sh
conda activate drl_nav

cd ${SCRATCH}/Algorithms
python RunPolicy.py -r 3 -c "sac_baseline_moving_platform_horizontal.yaml" -e "DRLNavigation" -t 20

echo "Copying Results Back"
mkdir -p ${PBS_O_WORKDIR}/Results/baseline_horizontal
cp -R ${SCRATCH}/Results/* ${PBS_O_WORKDIR}/Results/baseline_horizontal/

cd ${PBS_O_WORKDIR}
rm -r ${SCRATCH}

conda deactivate
module unload cuda/11.8