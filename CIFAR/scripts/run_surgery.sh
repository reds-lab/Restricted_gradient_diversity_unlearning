#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1
export MASTER_PORT=33333
ALG=RGD

# Arrays for alpha and lambda values
alphas=(1e-1) 
lambdas=(5) 

for alpha in "${alphas[@]}"
do
    for lambda in "${lambdas[@]}"
    do
        echo "Alpha: $alpha"
        echo "Lambda: $lambda"
        
        for CLASS in 0 1 2 3 4 5 6 7 8 9
        do
            CLASS_DIR=./training-runs/${ALG}_${alpha}_${lambda}/CLASS_$CLASS
            mkdir -p $CLASS_DIR
            echo "Current CLASS_DIR is ====================== $CLASS_DIR ======================"
            
            START_TIME=$(date +%s)
            python3 train.py --outdir=$CLASS_DIR --data=./data/cifar10-32x32.zip --cond=1 --arch=ddpmpp --batch=64 --transfer=./models/edm-cifar10-32x32-cond-vp.pkl --classes=$CLASS --unlearn-alg=$ALG --lr=1e-5 --duration=0.05 --ul-lambda=$lambda --ul-alpha=$alpha
            END_TIME=$(date +%s)
            DURATION=$((END_TIME - START_TIME))
            echo "Execution time for train.py: $DURATION seconds"
            
            SAVE_DIR=$CLASS_DIR/00000-cifar10-32x32-cond-ddpmpp-edm-gpus1-batch64-fp32-$ALG
            mkdir -p $SAVE_DIR/class/images/
            mkdir -p $SAVE_DIR/all_but_class/images/
            
            LATEST_PKL=$(ls $SAVE_DIR/network-snapshot-*.pkl | sort -V | tail -n 1)
            echo "Using latest network snapshot: $LATEST_PKL"
            
            python3 generate.py --network=$LATEST_PKL --outdir=$SAVE_DIR/class/ --class=$CLASS --seeds=0-49999 --batch=4500
            python3 generate.py --network=$LATEST_PKL --outdir=$SAVE_DIR/all_but_class/ --all_but_class=$CLASS --seeds=0-49999 --batch=4500
            
            python3 fid.py calc --images=$SAVE_DIR/class/images/ --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz --out_path=$CLASS_DIR/class_fid_$CLASS.txt --batch=4500
            python3 fid.py calc --images=$SAVE_DIR/all_but_class/images/ --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz --out_path=$CLASS_DIR/all_but_class_fid_$CLASS.txt --batch=4500
                        
            python3 ua.py --images=$SAVE_DIR/class/images/ --class=$CLASS --out_path=$CLASS_DIR/ua_clip_$CLASS.txt
            python3 ra.py --images=$SAVE_DIR/all_but_class/images/ --class_pkl=$SAVE_DIR/all_but_class/labels.pkl --out_path=$CLASS_DIR/ra_clip_$CLASS.txt

        done
        python3 compute_averages.py --input_dir=./training-runs/${ALG}_${alpha}_${lambda} --output_file=./training-runs/${ALG}_${alpha}_${lambda}/average_results_clip.txt

    done
done
