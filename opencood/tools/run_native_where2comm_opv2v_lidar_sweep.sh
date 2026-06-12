#!/usr/bin/env bash
set -euo pipefail

# Native Where2Comm baseline: use the checkpoint trained with the Where2Comm
# communication module, without inference-time selector replacement.

ROOT_DIR=${ROOT_DIR:-/home/zzh/projects/BlindMap}
PYTHON_BIN=${PYTHON_BIN:-/home/zzh/anaconda3/envs/coalign/bin/python}
GPU_ID=${GPU_ID:-0}
NUM_WORKERS=${NUM_WORKERS:-0}
MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/mplconfig}

MODEL_DIR=${MODEL_DIR:-/home/zzh/projects/BlindMap/opencood/logs/Where2comm_opv2v_lidar_pyramid_fair_2026_05_30_15_37_58_thre_0.01_add_noise_use_history}
RUN_MODEL_DIR=${RUN_MODEL_DIR:-${MODEL_DIR}}
RESULT_MODEL_DIR=${RESULT_MODEL_DIR:-${MODEL_DIR}}
MODAL=${MODAL:-0}
DET_RANGE=${DET_RANGE:-102.4,102.4}
BUDGETS=(${BUDGETS:-1 0.5 0.25 0.125 0.0625 0.03125 0.015625})

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR
export PYTHONUNBUFFERED=1

cd "${ROOT_DIR}"

log_dir="${RESULT_MODEL_DIR}/native_where2comm_budget_logs"
mkdir -p "${log_dir}"

{
  echo
  echo "############## native Where2Comm OPV2V LiDAR sweep: modal=${MODAL}, range=${DET_RANGE}, budgets=${BUDGETS[*]} ##############"
} >> "${RESULT_MODEL_DIR}/result.txt"

for budget in "${BUDGETS[@]}"; do
  safe_budget="${budget//./p}"
  log_file="${log_dir}/native_where2comm_lidar_comm_${safe_budget}MB.log"
  echo "[$(date '+%F %T')] native_where2comm_lidar: modal=${MODAL}, range=${DET_RANGE}, comm_volume_MB=${budget}" | tee -a "${log_file}"

  if ! "${PYTHON_BIN}" "${ROOT_DIR}/opencood/tools/inference.py" \
    --model_dir "${RUN_MODEL_DIR}" \
    --fusion_method intermediate \
    --modal "${MODAL}" \
    --range "${DET_RANGE}" \
    --comm_volume_MB "${budget}" \
    --note "_native_where2comm_lidar_${safe_budget}MB" \
    --num_workers "${NUM_WORKERS}" \
    --disable_vis \
    2>&1 | tee -a "${log_file}"; then
    echo "[$(date '+%F %T')] FAILED native_where2comm_lidar: comm_volume_MB=${budget}" | tee -a "${log_dir}/native_where2comm_lidar_FAILED.log"
  fi
  if [[ "${RUN_MODEL_DIR}" != "${RESULT_MODEL_DIR}" && -f "${RUN_MODEL_DIR}/result.txt" ]]; then
    tail -n 1 "${RUN_MODEL_DIR}/result.txt" >> "${RESULT_MODEL_DIR}/result.txt"
  fi
done
