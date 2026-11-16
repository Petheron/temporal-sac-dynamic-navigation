#!/bin/bash

#PBS -N tf_smoke
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=64GB
#PBS -l walltime=01:00:00
#PBS -q small_gpuq
#PBS -m abe
#PBS -M adam.cameron@student.uts.edu.au

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

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
python RunPolicy.py -r 1 -c "sac_transformer_SMOKETEST.yaml" -e "DRLNavigation" -t 20 -id 70 -n "_smoketest"

# ===== SAFEGUARDS START ======

RESULTS_DIR="${PBS_O_WORKDIR}/Results/${PBS_JOBNAME}_${PBS_JOBID%.*}"
mkdir -p ${RESULTS_DIR}
echo "Copying Results Back to ${RESULTS_DIR}"
cp -R ${SCRATCH}/Results/* ${RESULTS_DIR}/ 2>/dev/null

echo "===== What the Run Produced ====="
ls -lR ${SCRATCH}/Results/ || echo "NO RESULTS DIRECTORY AT ALL"

# ===== SAFEGUARDS END ======

cd ${PBS_O_WORKDIR}
rm -r ${SCRATCH}

conda deactivate
module unload cuda/11.8