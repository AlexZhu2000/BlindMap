#!/usr/bin/fish

set MODEL_DIR $argv[1]
set RANGE $argv[2]
set CONDA_ENV $argv[3]
set CUDA_VISIBLE_DEVICES $argv[4]
set MODAL $argv[5]

conda activate $CONDA_ENV

set CUDA "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Bandwidth ranges: 0: [1-10]Mbps, 1: [10-30]Mbps, 2: [30-50]Mbps, 3: [50-100]Mbps, 4: [100-200]Mbps
echo "Starting bandwidth simulation tests..."
echo "Bandwidth ranges:"
echo "  0: [1-10] Mbps (extremely low)"
echo "  1: [10-30] Mbps (low)"
echo "  2: [30-50] Mbps (medium)"
echo "  3: [50-100] Mbps (high)"
echo "  4: [100-200] Mbps (extremely high)"
echo ""

# Test each bandwidth range from 0 to 4
for BANDWIDTH in (seq 0 1 4)

    set CMD $CUDA

    set CMD "$CMD python opencood/tools/inference.py"

    set DELAY "--time_delay 0"
    set BANDWIDTH_ARG "--bandwidth $BANDWIDTH"
    set RANGE_ARG "--range $RANGE"
    set MODEL_DIR_ARG "--model_dir $MODEL_DIR"
    set MODAL_ARG "--modal $MODAL"
    set CMD "$CMD $MODEL_DIR_ARG $BANDWIDTH_ARG $RANGE_ARG $MODAL_ARG $DELAY"

    echo "=========================================="
    echo "Running with bandwidth range $BANDWIDTH"
    echo "Command: $CMD"
    echo "=========================================="
    eval $CMD

    if test $status -ne 0
        echo "Error: Command failed with bandwidth $BANDWIDTH"
        exit $status
    end

    echo ""
end

echo "All bandwidth simulation tests completed successfully!"

# Usage example:
# ./infer_bandwidth.fish /home/zzh/projects/BlindMap/opencood/logs/BlindMap_v2xset_camera_pyramid_2026_01_06_14_36_03_thre_0.01_add_noise_use_history 102.4,102.4 heal 1 1