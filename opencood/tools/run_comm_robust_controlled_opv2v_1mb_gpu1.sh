#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/zzh/projects/BlindMap}"
PYTHON_BIN="${PYTHON_BIN:-/home/zzh/anaconda3/envs/coalign/bin/python}"
MODEL_DIR="${MODEL_DIR:-/home/zzh/projects/BlindMap/opencood/logs/BlindMap_opv2v_m1m2_2025_12_23_19_23_52_thre_0.01_use_history*}"
GPU_ID="${GPU_ID:-1}"

MODAL="${MODAL:-0}"
DET_RANGE="${DET_RANGE:-102.4,102.4}"
COMM_VOLUME_MB="${COMM_VOLUME_MB:-1.0}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SEED="${SEED:-303}"
PACKET_SIZE="${PACKET_SIZE:-8}"
FUSION_METHOD="${FUSION_METHOD:-intermediate}"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}"
MAX_BATCHES_ARG=()
if [[ -n "${MAX_BATCHES:-}" ]]; then
  MAX_BATCHES_ARG=(--max_batches "${MAX_BATCHES}")
fi

LOG_DIR="${MODEL_DIR}/comm_robust_controlled_logs"
mkdir -p "${LOG_DIR}"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
SUMMARY_LOG="${LOG_DIR}/controlled_1mb_gpu${GPU_ID}_${RUN_TAG}.log"

BASE_ARGS=(
  --model_dir "${MODEL_DIR}"
  --fusion_method "${FUSION_METHOD}"
  --modal "${MODAL}"
  --range "${DET_RANGE}"
  --comm_volume_MB "${COMM_VOLUME_MB}"
  --num_workers "${NUM_WORKERS}"
  --packet_size "${PACKET_SIZE}"
  --seed "${SEED}"
  "${MAX_BATCHES_ARG[@]}"
)

run_case() {
  local name="$1"
  shift
  local case_log="${LOG_DIR}/${name}_${RUN_TAG}.log"

  {
    echo "[$(date '+%F %T')] START ${name}"
    echo "GPU_ID=${GPU_ID} MODAL=${MODAL} RANGE=${DET_RANGE} COMM_VOLUME_MB=${COMM_VOLUME_MB} NUM_WORKERS=${NUM_WORKERS} SEED=${SEED} PACKET_SIZE=${PACKET_SIZE}"
    echo "extra_args=$*"
  } | tee -a "${SUMMARY_LOG}"

  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" -u \
    "${ROOT_DIR}/opencood/tools/inference_comm_robust.py" \
    "${BASE_ARGS[@]}" \
    --setting_name "controlled_${name}" \
    --note "_controlled_${name}" \
    "$@" > "${case_log}" 2>&1

  echo "[$(date '+%F %T')] DONE ${name}; log=${case_log}" | tee -a "${SUMMARY_LOG}"
}

{
  echo "############## controlled communication robustness sweep ##############"
  echo "All cases use OPV2V test, modal=${MODAL}, range=${DET_RANGE}, comm_volume_MB=${COMM_VOLUME_MB}, packet_size=${PACKET_SIZE}, seed=${SEED}, num_workers=${NUM_WORKERS}."
  echo "Only the named network factor changes in each case; other simulator factors stay at the ideal defaults."
  echo "Run tag: ${RUN_TAG}"
} | tee -a "${MODEL_DIR}/comm_robust_result.txt" "${SUMMARY_LOG}"

run_case ideal \
  --packet_loss_prob 0.0 \
  --collab_dropout_prob 0.0 \
  --base_latency_ms 0.0 \
  --queue_delay_mean_ms 0.0 \
  --jitter_std_ms 0.0 \
  --max_retransmissions 0 \
  --loss_model iid

for loss_prob in 0.05 0.10 0.20 0.30; do
  loss_tag="${loss_prob/./p}"
  run_case "packet_loss_${loss_tag}" \
    --packet_loss_prob "${loss_prob}" \
    --collab_dropout_prob 0.0 \
    --base_latency_ms 0.0 \
    --queue_delay_mean_ms 0.0 \
    --jitter_std_ms 0.0 \
    --max_retransmissions 0 \
    --loss_model iid
done

for dropout_prob in 0.10 0.20 0.30; do
  dropout_tag="${dropout_prob/./p}"
  run_case "collab_dropout_${dropout_tag}" \
    --packet_loss_prob 0.0 \
    --collab_dropout_prob "${dropout_prob}" \
    --base_latency_ms 0.0 \
    --queue_delay_mean_ms 0.0 \
    --jitter_std_ms 0.0 \
    --max_retransmissions 0 \
    --loss_model iid
done

for delay_ms in 10 30 50; do
  run_case "queue_delay_${delay_ms}ms_deadline50ms" \
    --packet_loss_prob 0.0 \
    --collab_dropout_prob 0.0 \
    --base_latency_ms 0.0 \
    --queue_delay_mean_ms "${delay_ms}" \
    --jitter_std_ms 0.0 \
    --deadline_ms 50 \
    --max_retransmissions 0 \
    --loss_model iid
done

for jitter_ms in 5 10 20; do
  run_case "jitter_${jitter_ms}ms_deadline50ms" \
    --packet_loss_prob 0.0 \
    --collab_dropout_prob 0.0 \
    --base_latency_ms 0.0 \
    --queue_delay_mean_ms 0.0 \
    --jitter_std_ms "${jitter_ms}" \
    --deadline_ms 50 \
    --max_retransmissions 0 \
    --loss_model iid
done

run_case burst_loss_ge_default \
  --packet_loss_prob 0.0 \
  --collab_dropout_prob 0.0 \
  --base_latency_ms 0.0 \
  --queue_delay_mean_ms 0.0 \
  --jitter_std_ms 0.0 \
  --max_retransmissions 0 \
  --loss_model gilbert_elliott \
  --ge_good_to_bad 0.02 \
  --ge_bad_to_good 0.20 \
  --ge_good_loss 0.01 \
  --ge_bad_loss 0.50

echo "[$(date '+%F %T')] ALL DONE" | tee -a "${SUMMARY_LOG}"
