#! /bin/bash

export GPU_NUM=2
export CUDA_VISIBLE_DEVICES=0,1

dataset_list=("Cable" "Capacitor" "Casting" "Console" "Cylinder" "Electronics" "Groove" "Hemisphere" "Lens" "PCB_1" "PCB_2" "Ring" "Screw" "Wood")


for dataset_name in "${dataset_list[@]}"
do
	echo ""
	echo ""
	echo "------------------------------------------------------------------------------"
	echo "$dataset_name"
	export DATASET_NAME="$dataset_name"

      cp htc_swin_template.py htc_swin.py
	sed -i "s/DATASET_NAME/$dataset_name/g" htc_swin.py
	head -n 2 htc_swin.py

      bash ${CBNET_PATH}/tools/dist_test.sh \
             htc_swin.py \
             result/0.weight/${DATASET_NAME}/latest.pth \
             ${GPU_NUM} \
             --format-only \
             --options "jsonfile_prefix=./result/1.json/${DATASET_NAME}"
done
