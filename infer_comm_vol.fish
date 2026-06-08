#!/usr/bin/fish

set MODEL_DIR $argv[1]
set RANGE $argv[2]
set CONDA_ENV $argv[3]
set CUDA_VISIBLE_DEVICES $argv[4]
set MODAL $argv[5]

conda activate $CONDA_ENV

set CUDA "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# 初始通信量
set COMM_VOLUME 2

# 共执行 10 次：1, 1/2, 1/4, ..., 1/512
for i in (seq 1 1 10)

    set CMD $CUDA

    set CMD "$CMD python opencood/tools/inference.py"

    set DELAY "--time_delay 0"
    

    set COMM_THRE "--comm_volume_MB $COMM_VOLUME"
    set RANGE_ARG "--range $RANGE"
    set MODEL_DIR_ARG "--model_dir $MODEL_DIR"
    set MODAL_ARG "--modal $MODAL"
    set CMD "$CMD $MODEL_DIR_ARG $COMM_THRE $RANGE_ARG $MODAL_ARG $DELAY"

    echo "Running: $CMD"
    eval $CMD

    if test $status -ne 0
        exit $status
    end

    # COMM_VOLUME /= 2
    set COMM_VOLUME (math "$COMM_VOLUME / 2")

end

Run with comm_volume = 0
set CMD $CUDA
set CMD "$CMD python opencood/tools/inference.py"
set DELAY "--time_delay 0"
set COMM_THRE "--comm_volume_MB 0"
set RANGE_ARG "--range $RANGE"
set MODEL_DIR_ARG "--model_dir $MODEL_DIR"
set MODAL_ARG "--modal $MODAL"
set CMD "$CMD $MODEL_DIR_ARG $COMM_THRE $RANGE_ARG $MODAL_ARG $DELAY"

echo "Running with comm_volume=0: $CMD"
eval $CMD

if test $status -ne 0
    exit $status
end

# ./infer_comm_vol.fish /home/zzh/projects/BlindMap/opencood/logs/BlindMap_DAIR_camera_pyramid_2025_12_22_20_58_18_thre_0.01 102.4,102.4 heal 1 1