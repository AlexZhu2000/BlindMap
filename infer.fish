#!/usr/bin/fish
set MODEL_DIR $argv[1]
set RANGE $argv[2]
set CONDA_ENV $argv[3]
set CUDA_VISIBLE_DEVICES $argv[4]
set MODEL_NAME $argv[5]
conda activate $CONDA_ENV

set CUDA "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

for delay in (seq 1 1 5)
	set CMD $CUDA
	if test -n "$MODEL_NAME"
		set CMD "$CMD python opencood/tools/inference_comm_bp_plus.py --modal 0"
		set MODEL_NAME_ARG "--model_name $MODEL_NAME"
		set DELAY "--delay_time $delay"
	else
		set CMD "$CMD python opencood/tools/inference_zzh.py"
		set MODEL_NAME_ARG ""
		set DELAY "--time_delay $delay"
	end
	set COMM_THRE "--comm_volume_MB 1"
	set RANGE_ARG "--range $RANGE" 
	set MODEL_DIR_ARG "--model_dir $MODEL_DIR"
	set CMD "$CMD $MODEL_DIR_ARG $COMM_THRE $RANGE_ARG $MODEL_NAME_ARG $DELAY"
	echo $CMD
	eval $CMD	

       if test $status -ne 0
               exit $status
       end

 end
