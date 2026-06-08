#!/usr/bin/fish
set MODEL_DIR $argv[1]
set RANGE $argv[2]
set CONDA_ENV $argv[3]
set CUDA_VISIBLE_DEVICES $argv[4]

conda activate $CONDA_ENV

if test $status -ne 0
	exit $status
end

for i in (seq 1 5)
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python opencood/tools/inference.py --model_dir $MODEL_DIR --comm_volume_MB 1 --range $RANGE --time_delay $i

    if test $status -ne 0
        exit $status
    end

end


# for rot_std in (seq 0.1 0.1 0.8)
# 	CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python opencood/tools/inference_zzh.py --model_dir $MODEL_DIR --comm_volume_MB 1 --noise "0,$rot_std,0,0" --range $RANGE
	
# 	if test $status -ne 0
# 		exit $status
# 	end

# end
