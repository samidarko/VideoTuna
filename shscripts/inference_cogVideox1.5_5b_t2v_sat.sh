#! /bin/bash

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

environs="WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1"

run_cmd="$environs python scripts/inference_cogVideo_sat.py --base configs/005_cogvideox1.5/cogvideox1.5_5b_t2v.yaml configs/005_cogvideox1.5/inference1.5_5b_t2v.yaml --seed $RANDOM"

echo ${run_cmd}
eval ${run_cmd}

echo "DONE on `hostname`"