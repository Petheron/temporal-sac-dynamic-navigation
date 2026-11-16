#!/bin/bash

#PBS -N ls_h_t070_s4
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=32GB
#PBS -l walltime=48:00:00
#PBS -q small_gpuq
#PBS -m abe
#PBS -M adam.cameron@student.uts.edu.au

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

# ===== PER-RUN SETTINGS =====
# These two lines and "#PBS -N" above are all that differ between submission scripts
# CONFIG drives both the run and the marker file, so the two cannot drift
CONFIG="sac_ls_h_w10_t070_s4.yaml"
ENV_ID=904

SCRATCH="/scratch/${USER}_${PBS_JOBID%.*}"
RESULTS_DIR="${PBS_O_WORKDIR}/Results/${PBS_JOBNAME}_${PBS_JOBID%.*}"
mkdir -p ${SCRATCH}
mkdir -p ${RESULTS_DIR}

echo "===== JOB ${PBS_JOBNAME} (${PBS_JOBID}) ====="
echo "config  : ${CONFIG}"
echo "env_id  : ${ENV_ID}"
echo "scratch : ${SCRATCH}"
echo "results : ${RESULTS_DIR}"
echo "node    : $(hostname)"
echo "started : $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Clear scratch no matter how the job ends: normal exit, walltime, qdel or crash
# Nothing is deleted until it has been verified as copied out
cleanup() {
    echo "===== CLEANUP TRIGGERED ====="
    if [ -n "${WATCHER_PID:-}" ]; then
        kill ${WATCHER_PID} 2>/dev/null
    fi

    HAD_DATA=0
    COPY_OK=0

    if [ -d "${SCRATCH}/Results/SAC_DRLNavigation" ]; then
        HAD_DATA=1
        SRC="${SCRATCH}/Results/SAC_DRLNavigation"

        # Record which job, config and node produced this result set
        echo "job_name=${PBS_JOBNAME}"                      >  ${SRC}/job_marker.txt
        echo "job_id=${PBS_JOBID}"                          >> ${SRC}/job_marker.txt
        echo "config=${CONFIG}"                             >> ${SRC}/job_marker.txt
        echo "env_id=${ENV_ID}"                             >> ${SRC}/job_marker.txt
        echo "node=$(hostname)"                             >> ${SRC}/job_marker.txt
        echo "python_exit=${PY_STATUS:-killed_before_exit}"  >> ${SRC}/job_marker.txt
        echo "cleanup_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"     >> ${SRC}/job_marker.txt

        # Record the config values this run actually trained with
        # RunPolicy.py copies the config into the results dir, so read it there
        # Configs are CRLF, hence the \r strip
        record_field() {
            local found
            found=$(grep -m1 -E "^[[:space:]]*$1:" ${SRC}/*.yaml 2>/dev/null \
                    | sed 's/.*: *//' | tr -d '\r')
            echo "cfg_$1=${found:-<absent>}" >> ${SRC}/job_marker.txt
        }

        for field in observation_stacks sequence_length memory_size level_type \
                     platform_travel_time randomise_platform_phase seed \
                     num_test_eps num_action_repeat store_positions \
                     temporal_architecture transformer_n_layers \
                     transformer_n_heads; do
            record_field ${field}
        done

        # Checksum the algorithm sources so two runs can be proven to share code
        # The Sept bug ran Transformer configs through the LSTM path
        # Identical md5s across supposedly different architectures would show that
        for src_file in SAC.py SAC_LSTM.py SAC_GTrXL.py Encoders.py \
                        OffPolicyMain.py RunPolicy.py DataStructures.py; do
            if [ -f "${SCRATCH}/Algorithms/${src_file}" ]; then
                echo "md5_${src_file}=$(md5sum ${SCRATCH}/Algorithms/${src_file} | cut -d' ' -f1)" \
                    >> ${SRC}/job_marker.txt
            else
                echo "md5_${src_file}=<absent>" >> ${SRC}/job_marker.txt
            fi
        done

        # Checksum the Unity script assembly: this is what differs between builds
        # The launcher binary is ~15KB and does not change when the C# does
        # Located with find so the Managed path is not hardcoded
        ASSEMBLY=$(find ${SCRATCH}/Environments/DRLNavigation/Build/Linux \
                        -name "Assembly-CSharp.dll" -print -quit 2>/dev/null)
        if [ -n "${ASSEMBLY}" ] && [ -f "${ASSEMBLY}" ]; then
            echo "assembly_path=${ASSEMBLY#${SCRATCH}/}"              >> ${SRC}/job_marker.txt
            echo "md5_assembly=$(md5sum ${ASSEMBLY} | cut -d' ' -f1)" >> ${SRC}/job_marker.txt
            echo "assembly_mtime=$(stat -c %y ${ASSEMBLY})"           >> ${SRC}/job_marker.txt
        else
            echo "md5_assembly=<absent>" >> ${SRC}/job_marker.txt
        fi

        mkdir -p ${RESULTS_DIR}/SAC_DRLNavigation

        # Small files first, since SIGKILL can follow SIGTERM within seconds
        # Model dictionaries are ~235MB; results.npy is 17KB and matters more
        cp ${SRC}/results.npy ${SRC}/seeds.npy ${SRC}/job_marker.txt \
           ${SRC}/*.yaml ${RESULTS_DIR}/SAC_DRLNavigation/ 2>/dev/null

        if [ -s "${RESULTS_DIR}/SAC_DRLNavigation/results.npy" ]; then
            COPY_OK=1
            echo "METRICS SAFE: results.npy copied ($(stat -c%s ${RESULTS_DIR}/SAC_DRLNavigation/results.npy) bytes)"
        else
            echo "COPY FAILED: results.npy missing or empty after copy"
        fi

        # Now for the model and optimiser dictionaries, and the ONNX exports
        echo "Copying full results to ${RESULTS_DIR}"
        cp -R ${SCRATCH}/Results/* ${RESULTS_DIR}/ 2>/dev/null
        echo "===== What Came Back ====="
        ls -lR ${RESULTS_DIR}/ 2>/dev/null
    else
        echo "No Results Directory Found in Scratch (training may not have started)"
    fi

    # Delete scratch only if nothing needed saving, OR saving verifiably worked
    # Rationale: a stranded run can be recovered by hand, a deleted one cannot
    if [ "${HAD_DATA}" -eq 0 ] || [ "${COPY_OK}" -eq 1 ]; then
        rm -rf ${SCRATCH}
        echo "Scratch Directory Removed: ${SCRATCH}"
    else
        echo "SCRATCH DELIBERATELY RETAINED: ${SCRATCH}"
        echo "Data existed but nothing was verified as copied; requires manual recovery"
    fi

    echo "===== CLEANUP DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
}
trap cleanup EXIT

cd ${PBS_O_WORKDIR}
module load cuda/11.8

# Record the compute node's hardware for my thesis
# Whole block runs before training starts (costs well under 1s)
{
    echo "hostname=$(hostname)"
    echo "job_id=${PBS_JOBID}"
    echo "captured_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "===== nvidia-smi ====="
    nvidia-smi
    echo "===== GPU Query ====="
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
    echo "===== Compute Capability ====="
    nvidia-smi --query-gpu=compute_cap --format=csv
    echo "===== lscpu ====="
    lscpu
    echo "===== Memory ====="
    free -h
} > ${RESULTS_DIR}/node_hardware.txt 2>&1



cp -R Algorithms ${SCRATCH}/
cp -R Environments ${SCRATCH}/
mkdir ${SCRATCH}/Results

# Copy the metrics back every 10 minutes, independently of any trap
# This still works if SIGKILL lands before cleanup can run
# Checkpoints arrive far less often than this, so at most one is ever lost
# The loop sleeps, so it costs practically nothing against the ncpus limit
(
  while true; do
    sleep 600
    cp ${SCRATCH}/Results/SAC_DRLNavigation/results.npy \
       "${RESULTS_DIR}/results_PARTIAL.npy" 2>/dev/null
  done
) &
WATCHER_PID=$!

source ~/miniconda3/etc/profile.d/conda.sh
conda activate drl_nav

cd ${SCRATCH}/Algorithms
# -ng forces the null graphics device
python RunPolicy.py -r 1 -c "${CONFIG}" -e "DRLNavigation" -t 20 -id ${ENV_ID} -ng
PY_STATUS=$?

kill ${WATCHER_PID} 2>/dev/null
echo "python exited with status: ${PY_STATUS}"

cd ${PBS_O_WORKDIR}
conda deactivate
module unload cuda/11.8