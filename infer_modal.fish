#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
MODAL="opencood/tools/inference.py --model_dir /home/zzh/projects/BlindMap/opencood/logs/BlindMap_opv2v_m1m2_2025_12_23_19_23_52_thre_0.01_use_history"
for i in {1..4}
do
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python $MODAL --modal $i

    

done