#!/bin/bash

#PBS -N bl_dw_s2
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=64GB
#PBS -l walltime=08:00:00
#PBS -q small_gpuq
#PBS -m abe
#PBS -M adam.cameron@student.uts.edu.au

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

SCRATCH="/scratch/${USER}_${PBS_JOBID%.*}"
mkdir ${SCRATCH}

# Cleanup scratch no matter how the job ends (walltime, qdel, crash)
cleanup() {
    echo "===== CLEANUP TRIGGERED ====="

    # Only attempt copy-back if training produced results
    if [ -d "${SCRATCH}/Results/SAC_DRLNavigation" ]; then
        # Marker file to verify which job produced these results
        echo "job_name=${PBS_JOBNAME}" > ${SCRATCH}/Results/SAC_DRLNavigation/job_marker.txt
        echo "job_id=${PBS_JOBID}" >> ${SCRATCH}/Results/SAC_DRLNavigation/job_marker.txt
        echo "config=sac_baseline_moving_platform_depthwise_seed2.yaml" >> ${SCRATCH}/Results/SAC_DRLNavigation/job_marker.txt
        echo "completed=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> ${SCRATCH}/Results/SAC_DRLNavigation/job_marker.txt

        # Copy results to a job-ID stamped directory to guarantee no overlap
        RESULTS_DIR="${PBS_O_WORKDIR}/Results/${PBS_JOBNAME}_${PBS_JOBID%.*}"
        mkdir -p ${RESULTS_DIR}
        echo "Copying Results Back to ${RESULTS_DIR}"
        cp -R ${SCRATCH}/Results/* ${RESULTS_DIR}/

        # Verify copy succeeded by checking results.npy exists and has nonzero size
        if [ -s "${RESULTS_DIR}/SAC_DRLNavigation/results.npy" ]; then
            echo "COPY SUCCESSFUL: results.npy copied ($(stat -c%s ${RESULTS_DIR}/SAC_DRLNavigation/results.npy) bytes)"
        else
            echo "COPY FAILED: results.npy missing or empty after copy (job may have been killed before training produced output)"
        fi
    else
        echo "No Results Directory Found in Scratch (training may not have started or produced output)"
    fi

    rm -rf ${SCRATCH}
    echo "Scratch Directory Removed: ${SCRATCH}"
}
trap cleanup EXIT

cd ${PBS_O_WORKDIR}
module load cuda/11.8

cp -R Algorithms ${SCRATCH}/
cp -R Environments ${SCRATCH}/
mkdir ${SCRATCH}/Results

source ~/miniconda3/etc/profile.d/conda.sh
conda activate drl_nav

cd ${SCRATCH}/Algorithms
python RunPolicy.py -r 1 -c "sac_baseline_moving_platform_depthwise_seed2.yaml" -e "DRLNavigation" -t 20 -id 40

cd ${PBS_O_WORKDIR}
conda deactivate
module unload cuda/11.8