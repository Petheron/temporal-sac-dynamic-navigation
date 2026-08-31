#!/bin/bash

#PBS -N bl_hz_s1
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=64GB
#PBS -l walltime=08:00:00
#PBS -q small_gpuq
#PBS -m abe
#PBS -M adam.cameron@student.uts.edu.au

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

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
python RunPolicy.py -r 1 -c "sac_baseline_moving_platform_horizontal_seed1.yaml" -e "DRLNavigation" -t 20 -id 22

echo "Copying results back"
mkdir -p ${PBS_O_WORKDIR}/Results/bl_hz_s1
cp -R ${SCRATCH}/Results/* ${PBS_O_WORKDIR}/Results/bl_hz_s1/

cd ${PBS_O_WORKDIR}
rm -r ${SCRATCH}

conda deactivate
module unload cuda/11.8