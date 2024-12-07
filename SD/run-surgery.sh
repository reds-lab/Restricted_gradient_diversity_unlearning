#!/bin/sh

export CUDA_VISIBLE_DEVICES=0
export MASTER_PORT=33333

# run 
ALPHA=1.6 #1.6
M=1.1 # 1.5
Forget='y_c_train' # 'y_c_train_400' # y_c_train
Retain='y_train' # 'y_train_400' # y_train

echo "ALPHA: $ALPHA" ## descent weight control
echo "M: $M"

PROPOSED_PATH="proposed-$Forget-$Retain-alpha${ALPHA}-m${M}"
SAVE_PATH='./results/'

# python3 train-scripts/nsfw_RGD.py --train_method 'full' --truncate --surgery --proposed_path $PROPOSED_PATH --forget $Forget --retain $Retain --alpha $ALPHA --m $M
# python3 eval-scripts/evaluation-batchwise.py --model_name 'compvis-nsfw-method_full-lr_1e-05' --proposed --proposed_path $PROPOSED_PATH --forget $Forget --retain $Retain --save_path $SAVE_PATH
python3 eval-scripts/evaluation-i2p-batchwise.py --model_name 'compvis-nsfw-method_full-lr_1e-05' --proposed --proposed_path $PROPOSED_PATH --forget $Forget --retain $Retain --save_path $SAVE_PATH
python3 eval-scripts/nudenet-classes-json.py --model_name 'compvis-nsfw-method_full-lr_1e-05' --proposed --proposed_path $PROPOSED_PATH --forget $Forget --retain $Retain --save_path $SAVE_PATH

